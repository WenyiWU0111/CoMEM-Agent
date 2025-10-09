# websites domain
import os

REDDIT = os.environ.get("REDDIT", "")
SHOPPING = os.environ.get("SHOPPING", "")
GITLAB = os.environ.get("GITLAB", "")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
MAP = os.environ.get("MAP", "")
HOMEPAGE = os.environ.get("HOMEPAGE", "")
CLASSIFIEDS = os.environ.get("CLASSIFIEDS", "")
TWITTER = os.environ.get("TWITTER", "")

assert (
    REDDIT
    and SHOPPING
    and GITLAB
    and WIKIPEDIA
    and MAP
    and CLASSIFIEDS
    and HOMEPAGE
), (
    f"Please setup the URLs to each site. Current: "
    + f"Reddit: {REDDIT}"
    + f"Shopping: {SHOPPING}"
    + f"Gitlab: {GITLAB}"
    + f"Wikipedia: {WIKIPEDIA}"
    + f"Map: {MAP}"
    + f"Classifieds: {CLASSIFIEDS}"
    + f"Twitter: {TWITTER}"
    + f"Homepage: {HOMEPAGE}"
)


ACCOUNTS = {
}

URL_MAPPINGS = {
    REDDIT: "https://www.reddit.com/",
    SHOPPING: "https://www.amazon.com/",
    GITLAB: "https://gitlab.com/",
    WIKIPEDIA: "https://www.wikipedia.org/",
    MAP: "https://www.openstreetmap.org/",
    CLASSIFIEDS: "https://www.amazon.com/",
    HOMEPAGE: "https://www.wikipedia.org/",
    TWITTER: "https://twitter.com/",
}
