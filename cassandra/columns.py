class GameDfColumns:
    TEAM1_WIN_PROB = "team1_win_prob"
    TEAM1_WIN = "team1_win"
    TEAM1_MOV = "team1_mov"
    SPREAD = "spread"
    # The prediction markets' no-vig home win probability at their close, when
    # a caller has attached one (`markets.close_probabilities`). Not something
    # a prediction carries: the replay never sees it.
    MARKET_HOME_PROB = "market_home_prob"
