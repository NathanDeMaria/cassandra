"""How far the away team came, from a hand-entered table of home venues.

Naive on purpose, and the naivety is the table. Nothing in the stack knows
where a team plays: `endgame.types.Game` has no venue, and
`call_it_what_you_want`'s `Team` and `TeamClassification` carry names,
divisions and conferences and no location. So the coordinates below were
typed in by hand.

Read them as **approximate campus coordinates, not surveyed stadium
locations**, and verify before anything is built on them. They are good to
about a city, which is all a travel distance needs -- the question is
whether a team came 200 miles or 2,000, and a kilometre of error in the
venue changes no answer here. What they are *not* is authoritative: a
program that plays off-campus is filed at its campus (UCLA at the Rose Bowl
is the obvious one), and a school that moved stadiums has one entry for its
whole history.

Coverage is FBS only, because that is the population any travel feature
would serve and it is the one whose venues are stable enough to hard-code.
A team not in the table has no distance, and `distance_km` says so with
None rather than guessing -- see `cassandra.residuals`, where an unknown
slice is dropped rather than pooled into a wrong one.

Neutral-site games are the other gap and cannot be closed from here: the
`Game` says a game was neutral and never says where it was, so a bowl in
Miami and one in Pasadena are the same record. Those are excluded rather
than charged to the home team's venue.

What the distance was measured to be worth, and why it is pinned at 0
---------------------------------------------------------------------

`MatchupAdjustments.travel_points` reads this as rating points per 1,000 km,
added to the home side -- a home field advantage that grows with the trip.
Measured on 2026-09-12 against `glicko_full`'s current fit with the term
off, binning the win-probability residual by the away side's trip:

    ncaafb, 14,721 games          nfl, 6,538 games
    <250 km     +0.008            +0.086 (n=182)
    250-500     +0.010            +0.042
    500-1k      +0.017            +0.049
    1k-1.5k     +0.012            +0.044
    1.5k-2.5k   +0.031            +0.059
    2.5k+       +0.042 (n=681)    +0.072

ncaafb's runs about linear, ~0.011 of win probability per 1,000 km, and of
five framings tried -- linear, time zones crossed, sqrt, a step at 1,500 km,
and the six bucket means -- linear and time zones are the best and within
noise of each other (t = 3.5 and 3.6; brier gain over a refit home
advantage 0.00015). It is symmetric in direction: west by one, two, three
zones reads +0.027, +0.042, +0.067 and east +0.019, +0.048, +0.058, so it
is not a body-clock effect; and binning by the away side's local kickoff
hour finds nothing, with the before-10am bucket the *smallest* at +0.008.
Distance, linear, and small. nfl has nothing at all: every form is t ~ 1
and the buckets are flat between +0.04 and +0.07.

So the framing was not the problem, and the term is pinned at 0 in every
football config because 0.00015 on one league is not worth a search
dimension -- a `glicko_full` search put it against the zero bound in nfl
with 58% of its best probes crowding the edge. The ncaafb 0.00015 is real
and is the one cost of the pin; a config can re-open it by moving
`travel_advantage` back under `parameters`.
"""

import math
from collections.abc import Sequence

#: Where one team plays. A bare pair is a team that has always played in the
#: same place, which is nearly all of them. A list of `(year, lat, lon)` steps
#: is a franchise that moved: the coordinates are the ones in effect from that
#: year on, so the seasons before a relocation are measured from the old city.
#:
#: The same two shapes, for the same reason, as `predictor.base_predictor`'s
#: `Anchor` -- most entries never change and writing every one of them a
#: history would bury the three that did.
Venue = tuple[float, float] | Sequence[tuple[int, float, float]]

#: (latitude, longitude) per team, in the canonical names the replay uses --
#: `TeamNamer.for_league("ncaafb")` output, so these join straight onto a
#: predictions frame without a second naming vocabulary.
#:
#: Hand-entered and approximate; see the module docstring.
VENUES: dict[str, Venue] = {
    "Air Force Falcons": (38.99, -104.86),
    "Akron Zips": (41.08, -81.52),
    "Alabama Crimson Tide": (33.21, -87.55),
    "App State Mountaineers": (36.21, -81.68),
    "Arizona State Sun Devils": (33.42, -111.93),
    "Arizona Wildcats": (32.23, -110.95),
    "Arkansas Razorbacks": (36.07, -94.17),
    "Arkansas State Red Wolves": (35.84, -90.70),
    "Army Black Knights": (41.39, -73.96),
    "Auburn Tigers": (32.60, -85.49),
    "BYU Cougars": (40.25, -111.65),
    "Ball State Cardinals": (40.20, -85.41),
    "Baylor Bears": (31.55, -97.11),
    "Boise State Broncos": (43.60, -116.20),
    "Boston College Eagles": (42.34, -71.17),
    "Bowling Green Falcons": (41.37, -83.65),
    "Buffalo Bulls": (43.00, -78.79),
    "California Golden Bears": (37.87, -122.25),
    "Central Michigan Chippewas": (43.59, -84.77),
    "Charlotte 49ers": (35.31, -80.73),
    "Cincinnati Bearcats": (39.13, -84.52),
    "Clemson Tigers": (34.68, -82.84),
    "Coastal Carolina Chanticleers": (33.79, -79.01),
    "Colorado Buffaloes": (40.01, -105.27),
    "Colorado State Rams": (40.57, -105.08),
    "Delaware Blue Hens": (39.68, -75.75),
    "Duke Blue Devils": (36.00, -78.94),
    "East Carolina Pirates": (35.61, -77.37),
    "Eastern Michigan Eagles": (42.24, -83.62),
    "Florida Atlantic Owls": (26.37, -80.10),
    "Florida Gators": (29.65, -82.34),
    "Florida International Panthers": (25.76, -80.37),
    "Florida State Seminoles": (30.44, -84.30),
    "Fresno State Bulldogs": (36.81, -119.75),
    "Georgia Bulldogs": (33.95, -83.38),
    "Georgia Southern Eagles": (32.42, -81.78),
    "Georgia State Panthers": (33.75, -84.39),
    "Georgia Tech Yellow Jackets": (33.78, -84.40),
    "Hawai'i Rainbow Warriors": (21.30, -157.82),
    "Houston Cougars": (29.72, -95.34),
    "Idaho Vandals": (46.73, -117.00),
    "Illinois Fighting Illini": (40.10, -88.24),
    "Indiana Hoosiers": (39.17, -86.52),
    "Iowa Hawkeyes": (41.66, -91.55),
    "Iowa State Cyclones": (42.03, -93.65),
    "Jacksonville State Gamecocks": (33.82, -85.77),
    "James Madison Dukes": (38.44, -78.87),
    "Kansas Jayhawks": (38.96, -95.25),
    "Kansas State Wildcats": (39.19, -96.58),
    "Kennesaw State Owls": (34.04, -84.58),
    "Kent State Golden Flashes": (41.15, -81.34),
    "Kentucky Wildcats": (38.03, -84.50),
    "LSU Tigers": (30.41, -91.18),
    "Liberty Flames": (37.35, -79.18),
    "Louisiana Ragin' Cajuns": (30.21, -92.02),
    "Louisiana Tech Bulldogs": (32.53, -92.65),
    "Louisville Cardinals": (38.21, -85.76),
    "Marshall Thundering Herd": (38.42, -82.42),
    "Maryland Terrapins": (38.99, -76.94),
    "Massachusetts Minutemen": (42.39, -72.53),
    "Memphis Tigers": (35.12, -89.94),
    "Miami (OH) RedHawks": (39.51, -84.73),
    "Miami Hurricanes": (25.96, -80.24),
    "Michigan State Spartans": (42.73, -84.48),
    "Michigan Wolverines": (42.27, -83.74),
    "Middle Tennessee Blue Raiders": (35.85, -86.37),
    "Minnesota Golden Gophers": (44.98, -93.23),
    "Mississippi State Bulldogs": (33.46, -88.79),
    "Missouri State Bears": (37.20, -93.28),
    "Missouri Tigers": (38.94, -92.33),
    "NC State Wolfpack": (35.78, -78.68),
    "Navy Midshipmen": (38.98, -76.49),
    "Nebraska Cornhuskers": (40.82, -96.71),
    "Nevada Wolf Pack": (39.54, -119.82),
    "New Mexico Lobos": (35.08, -106.62),
    "New Mexico State Aggies": (32.28, -106.75),
    "North Carolina Tar Heels": (35.91, -79.05),
    "North Texas Mean Green": (33.21, -97.15),
    "Northern Illinois Huskies": (41.93, -88.77),
    "Northwestern Wildcats": (42.06, -87.69),
    "Notre Dame Fighting Irish": (41.70, -86.24),
    "Ohio Bobcats": (39.32, -82.10),
    "Ohio State Buckeyes": (40.00, -83.02),
    "Oklahoma Sooners": (35.21, -97.44),
    "Oklahoma State Cowboys": (36.13, -97.07),
    "Old Dominion Monarchs": (36.89, -76.31),
    "Ole Miss Rebels": (34.36, -89.54),
    "Oregon Ducks": (44.06, -123.07),
    "Oregon State Beavers": (44.56, -123.28),
    "Penn State Nittany Lions": (40.81, -77.86),
    "Pittsburgh Panthers": (40.44, -80.02),
    "Purdue Boilermakers": (40.43, -86.92),
    "Rice Owls": (29.72, -95.40),
    "Rutgers Scarlet Knights": (40.52, -74.46),
    "SMU Mustangs": (32.84, -96.78),
    "Sam Houston Bearkats": (30.71, -95.55),
    "San Diego State Aztecs": (32.78, -117.07),
    "San José State Spartans": (37.34, -121.88),
    "South Alabama Jaguars": (30.70, -88.18),
    "South Carolina Gamecocks": (34.00, -81.03),
    "South Florida Bulls": (28.06, -82.41),
    "Southern Miss Golden Eagles": (31.33, -89.33),
    "Stanford Cardinal": (37.43, -122.17),
    "Syracuse Orange": (43.04, -76.14),
    "TCU Horned Frogs": (32.71, -97.36),
    "Temple Owls": (39.98, -75.16),
    "Tennessee Volunteers": (35.95, -83.93),
    "Texas A&M Aggies": (30.61, -96.34),
    "Texas Longhorns": (30.28, -97.73),
    "Texas State Bobcats": (29.89, -97.94),
    "Texas Tech Red Raiders": (33.58, -101.87),
    "Toledo Rockets": (41.66, -83.61),
    "Troy Trojans": (31.80, -85.95),
    "Tulane Green Wave": (29.94, -90.12),
    "Tulsa Golden Hurricane": (36.15, -95.94),
    "UAB Blazers": (33.50, -86.81),
    "UCF Knights": (28.60, -81.20),
    "UCLA Bruins": (34.07, -118.44),
    "UConn Huskies": (41.81, -72.25),
    "UL Monroe Warhawks": (32.53, -92.07),
    "UNLV Rebels": (36.11, -115.14),
    "USC Trojans": (34.02, -118.29),
    "UTEP Miners": (31.77, -106.50),
    "UTSA Roadrunners": (29.58, -98.62),
    "Utah State Aggies": (41.74, -111.81),
    "Utah Utes": (40.76, -111.85),
    "Vanderbilt Commodores": (36.14, -86.80),
    "Virginia Cavaliers": (38.03, -78.51),
    "Virginia Tech Hokies": (37.23, -80.42),
    "Wake Forest Demon Deacons": (36.13, -80.28),
    "Washington Huskies": (47.65, -122.30),
    "Washington State Cougars": (46.73, -117.16),
    "West Virginia Mountaineers": (39.65, -79.95),
    "Western Kentucky Hilltoppers": (36.99, -86.45),
    "Western Michigan Broncos": (42.28, -85.61),
    "Wisconsin Badgers": (43.07, -89.41),
    "Wyoming Cowboys": (41.31, -105.58),
    # ---- nfl. Names are the bare nicknames the replay uses for this league.
    #
    # Three franchises moved inside the window cassandra replays (1999 on) and
    # carry their history rather than a single point. Getting these wrong is
    # not a rounding error: the Rams' two homes are 2,500km apart, and a
    # single coordinate would misprice every trip they took for seventeen
    # seasons.
    "bears": (41.86, -87.62),
    "bengals": (39.10, -84.52),
    "bills": (42.77, -78.79),
    "broncos": (39.74, -105.02),
    "browns": (41.51, -81.70),
    "buccaneers": (27.98, -82.50),
    "cardinals": (33.53, -112.26),
    "chargers": ((1999, 32.78, -117.12), (2017, 33.95, -118.34)),
    "chiefs": (39.05, -94.48),
    "colts": (39.76, -86.16),
    "commanders": (38.91, -76.86),
    "cowboys": (32.75, -97.09),
    "dolphins": (25.96, -80.24),
    "eagles": (39.90, -75.17),
    "falcons": (33.76, -84.40),
    "giants": (40.81, -74.07),
    "jaguars": (30.32, -81.64),
    "jets": (40.81, -74.07),
    "lions": (42.34, -83.05),
    "niners": (37.40, -121.97),
    "packers": (44.50, -88.06),
    "panthers": (35.23, -80.85),
    "patriots": (42.09, -71.26),
    "raiders": ((1999, 37.75, -122.20), (2020, 36.09, -115.18)),
    "rams": ((1999, 38.63, -90.19), (2016, 33.95, -118.34)),
    "ravens": (39.28, -76.62),
    "saints": (29.95, -90.08),
    "seahawks": (47.60, -122.33),
    "steelers": (40.45, -80.02),
    "texans": (29.68, -95.41),
    "titans": (36.17, -86.77),
    "vikings": (44.97, -93.26),
}

_EARTH_RADIUS_KM = 6371.0


def venue(team: str, season: int | None = None) -> tuple[float, float] | None:
    """Where this team played in `season`, or None if the table has no entry.

    Clamped at both ends, the way `anchor_in` is: before the first step is
    the same answer as the first step, and after the last is the last. A
    `season` of None reads as "wherever it plays now", which is what a
    caller with no date wants.
    """
    found = VENUES.get(team)
    if found is None:
        return None
    first = found[0]
    if isinstance(first, (int, float)):
        # A bare (lat, lon): a team that has always played in one place.
        second = found[1]
        assert isinstance(second, (int, float))
        return (float(first), float(second))
    steps = [step for step in found if not isinstance(step, (int, float))]
    _, lat, lon = steps[0]
    if season is not None:
        for step_year, step_lat, step_lon in steps:
            if step_year > season:
                break
            lat, lon = step_lat, step_lon
    return (float(lat), float(lon))


def distance_km(away: str, home: str, season: int | None = None) -> float | None:
    """Great-circle km the away team travelled, or None if either is unknown.

    None rather than 0 for a missing venue: 0 is "they were already there",
    which is a claim, and an absent team is not one.

    Great-circle rather than road or flight distance because the ordering is
    what matters -- a bucketed axis only needs to know that Hawai'i to
    Storrs is a long way and Duke to NC State is not.

    `season` picks the right home for a franchise that moved. Without one
    every relocation is measured from where the team plays today, which is
    wrong by 2,500km for every Rams road trip before 2016.
    """
    start, end = venue(away, season), venue(home, season)
    if start is None or end is None:
        return None
    lat1, lon1 = math.radians(start[0]), math.radians(start[1])
    lat2, lon2 = math.radians(end[0]), math.radians(end[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))
