# GitHub Actions authenticates by OIDC rather than a long-lived access key.
# Two roles, not one: `terraform plan` executes provider code and runs on every
# branch and PR, so it must not be able to reach credentials that can apply.
#
# Same shape as endgame/jobs/oidc.tf and aws-batch-optimization's -- each stack
# creates and owns the roles its own workflow assumes, rather than one repo
# minting roles for the others. The trust policy is per-repository, so there's
# nothing to share and nothing to keep in sync.

locals {
  owner = split("/", var.github_repository)[0]
  name  = split("/", var.github_repository)[1]

  # The ID-qualified subject GitHub actually issues. IDs rather than names is
  # the point of the format: a repository can be renamed, but its ID can't be
  # taken over by someone else claiming the old name.
  subject_repo = "repo:${local.owner}@${var.github_owner_id}/${local.name}@${var.github_repository_id}"

  # Kept alongside it so the policy still works if an account is ever issuing
  # the older name-only subjects. Both forms are exact, so listing both widens
  # nothing -- GitHub signs the claim, it can't be spoofed.
  subject_repo_legacy = "repo:${var.github_repository}"

  # IAM permits exactly one provider per URL per account, and invisible-string
  # creates it. So this stack expects to find it rather than making a second
  # one, which would fail with EntityAlreadyExists.
  oidc_provider_arn = var.create_oidc_provider ? (
    aws_iam_openid_connect_provider.github[0].arn
    ) : (
    "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"
  )

  iam_prefix = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}"

  # Everything this stack creates is named `cassandra-*`: the job role and its
  # inline policy, and the two CI roles below. Scoping the apply role's IAM
  # grant to that prefix is what stops a compromised workflow minting itself an
  # admin role.
  managed_role_arns = ["${local.iam_prefix}:role/${var.resource_name_prefix}-*"]

  # PassRole is wider than the roles this stack owns, and has to be. A Batch
  # job definition is created with an execution role, and an EventBridge
  # schedule with a scheduler role -- both of those live in the shared stack
  # and are only referenced here. Creating either resource is a PassRole on a
  # role this stack will never manage.
  passable_role_arns = concat(
    local.managed_role_arns,
    [for name in var.shared_role_names : "${local.iam_prefix}:role/${name}"],
  )
}

resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_oidc_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = []
}

# ------------------------------------------------------------------------------
# Trust policies
#
# The `sub` claim's shape depends on the event, which is the easy thing to get
# wrong: a branch push is `repo:owner/name:ref:refs/heads/<branch>`, but a pull
# request is `repo:owner/name:pull_request` with no ref at all. A plan role
# trusting only `ref:refs/heads/*` therefore fails on every PR.
# ------------------------------------------------------------------------------

data "aws_iam_policy_document" "plan_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "${local.subject_repo}:ref:refs/heads/*",
        "${local.subject_repo}:pull_request",
        "${local.subject_repo_legacy}:ref:refs/heads/*",
        "${local.subject_repo_legacy}:pull_request",
      ]
    }
  }
}

data "aws_iam_policy_document" "apply_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # StringLike only because two exact alternatives are listed; neither
    # contains a wildcard, so "main-hotfix" still cannot match. Keeping the
    # branch segment literal is what stops any branch merely starting with
    # "main" from gaining apply rights.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "${local.subject_repo}:ref:refs/heads/main",
        "${local.subject_repo_legacy}:ref:refs/heads/main",
      ]
    }
  }
}

# ------------------------------------------------------------------------------
# Plan role: read everything, write nothing except the state lock.
# ------------------------------------------------------------------------------

resource "aws_iam_role" "ci_plan" {
  name               = "${var.resource_name_prefix}-ci-plan"
  description        = "terraform plan from any branch or PR of ${var.github_repository}"
  assume_role_policy = data.aws_iam_policy_document.plan_assume_role.json
}

# ReadOnlyAccess is also what lets the `terraform_remote_state` data source in
# main.tf read the Batch stack's state object -- a data source read takes no
# lock, so plain s3:GetObject covers it and no extra grant is needed.
resource "aws_iam_role_policy_attachment" "ci_plan_readonly" {
  role       = aws_iam_role.ci_plan.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/ReadOnlyAccess"
}

# Plan reads state and takes the lock, so it needs writes on the lock object
# even though it changes no infrastructure. Scoped to this stack's key: the
# Batch stack's state is readable (above) but not writable from here.
data "aws_iam_policy_document" "terraform_state" {
  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = ["arn:${data.aws_partition.current.partition}:s3:::${var.state_bucket}"]
  }

  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:s3:::${var.state_bucket}/${var.state_key_prefix}*",
    ]
  }
}

resource "aws_iam_policy" "terraform_state" {
  name        = "${var.resource_name_prefix}-terraform-state"
  description = "Read/write this stack's terraform state and its lock file"
  policy      = data.aws_iam_policy_document.terraform_state.json
}

resource "aws_iam_role_policy_attachment" "ci_plan_state" {
  role       = aws_iam_role.ci_plan.name
  policy_arn = aws_iam_policy.terraform_state.arn
}

# ------------------------------------------------------------------------------
# Apply role: main only.
# ------------------------------------------------------------------------------

resource "aws_iam_role" "ci_apply" {
  name               = "${var.resource_name_prefix}-ci-apply"
  description        = "terraform apply from main of ${var.github_repository}"
  assume_role_policy = data.aws_iam_policy_document.apply_assume_role.json
}

# PowerUserAccess covers Batch, EventBridge Scheduler, SNS and S3, and
# explicitly denies IAM. The IAM this stack does need is added below, scoped by
# name prefix, rather than by attaching IAMFullAccess.
resource "aws_iam_role_policy_attachment" "ci_apply_poweruser" {
  role       = aws_iam_role.ci_apply.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/PowerUserAccess"
}

data "aws_iam_policy_document" "ci_apply_iam" {
  statement {
    sid    = "ManageOwnRoles"
    effect = "Allow"
    actions = [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:GetRole",
      "iam:UpdateRole",
      "iam:UpdateRoleDescription",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:ListRoleTags",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:ListAttachedRolePolicies",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:GetRolePolicy",
      "iam:ListRolePolicies",
      "iam:UpdateAssumeRolePolicy",
    ]
    resources = local.managed_role_arns
  }

  statement {
    sid    = "ManageOwnPolicies"
    effect = "Allow"
    actions = [
      "iam:CreatePolicy",
      "iam:DeletePolicy",
      "iam:GetPolicy",
      "iam:ListPolicyVersions",
      "iam:CreatePolicyVersion",
      "iam:DeletePolicyVersion",
      "iam:GetPolicyVersion",
      "iam:TagPolicy",
      "iam:UntagPolicy",
      "iam:ListEntitiesForPolicy",
    ]
    resources = ["${local.iam_prefix}:policy/${var.resource_name_prefix}-*"]
  }

  # See `passable_role_arns`: the execution and scheduler roles belong to the
  # shared stack, and creating a job definition or a schedule passes them.
  statement {
    sid       = "PassOwnAndSharedRoles"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = local.passable_role_arns
  }

  # Reading any role is needed for plan-time refresh of things this stack
  # references but doesn't own.
  statement {
    sid    = "ReadIam"
    effect = "Allow"
    actions = [
      "iam:ListRoles",
      "iam:ListPolicies",
      "iam:GetOpenIDConnectProvider",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "ci_apply_iam" {
  name        = "${var.resource_name_prefix}-ci-apply-iam"
  description = "IAM management scoped to ${var.resource_name_prefix}-* roles and policies"
  policy      = data.aws_iam_policy_document.ci_apply_iam.json
}

resource "aws_iam_role_policy_attachment" "ci_apply_iam" {
  role       = aws_iam_role.ci_apply.name
  policy_arn = aws_iam_policy.ci_apply_iam.arn
}

resource "aws_iam_role_policy_attachment" "ci_apply_state" {
  role       = aws_iam_role.ci_apply.name
  policy_arn = aws_iam_policy.terraform_state.arn
}
