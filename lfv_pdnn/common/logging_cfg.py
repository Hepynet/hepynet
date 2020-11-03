import logging

logging.basicConfig(
    #format="%(asctime)s [%(levelname)s] %(message)s (in %(filename)s:%(lineno)d)",
    format="[%(levelname)s] %(message)s (in %(filename)s:%(lineno)d)",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,  # debug, info, warn, error, fatal
)
