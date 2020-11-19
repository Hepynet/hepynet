import logging

logging.basicConfig(
    format="[%(levelname)s] %(message)s (in %(filename)s:%(lineno)d)",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,  # debug, info, warn, error, fatal
)
