import argparse
import logging
import os
import time

logger = logging.getLogger("hepynet")
# Show tensorflow warnings and errors only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def execute():
    """Enterpoint function of hepynet"""
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_configs", nargs="*", action="store")
    parser.add_argument(
        "-d",
        "--debug",
        required=False,
        help="run in debug mode",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--resume",
        required=False,
        help="resume (tune) job",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--time",
        required=False,
        help="display time",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        help="verbose debug information",
        action="store_true",
    )
    args = parser.parse_args()

    # check input
    if not args.yaml_configs:
        parser.print_help()
        exit()
    else:
        # set logging level
        if args.debug:  # DEBUG, INFO, WARNING, ERROR, CRITICAL
            logging.getLogger("hepynet").setLevel(logging.DEBUG)
        else:
            logging.getLogger("hepynet").setLevel(logging.INFO)
        # set logging format
        logging_format = "%(levelname)s %(message)s"
        if args.verbose:
            logging_format += " @ %(filename)s:%(lineno)d - %(funcName)s"
        if args.time:
            logging_format = "%(asctime)s,%(msecs)03d " + logging_format
        logging.basicConfig(
            format=logging_format, datefmt="%Y-%m-%d:%H:%M:%S",
        )

        from hepynet.main import job_executor

        for yaml_cfg in args.yaml_configs:
            logger.info("#" * 80)
            logger.info(f"Executing: {yaml_cfg}")
            job_start_time = time.perf_counter()
            executor = job_executor.job_executor(yaml_cfg)
            executor.execute_jobs(resume=args.resume)
            job_end_time = time.perf_counter()
            time_consumed = job_end_time - job_start_time
            time_consumed_str = time.strftime(
                "%H:%M:%S", time.gmtime(time_consumed)
            )
            logger.info(f"Time consumed: {time_consumed_str}")
        logger.info("#" * 80)
        logger.info("Done!")
        logger.info("#" * 80)
