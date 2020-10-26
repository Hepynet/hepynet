import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)d [%(levelname)s] %(message)s (in %(filename)s:%(lineno)d)",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,  # debug, info, warn, error, fatal
)
