import argparse
import logging
import os
import sys

from lfv_pdnn.main import job_executor

# Show tensorflow warnings and errors only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_configs", nargs="*", action="store")
    parser.add_argument(
        "-d", "--debug", required=False, help="run in debug mode", action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        help="verbose debug infomation",
        action="store_true",
    )
    args = parser.parse_args()

    # check input
    if not args.yaml_configs:
        parser.print_help()
        exit()
    else:
        # set debug level
        if args.debug:  # DEBUG, INFO, WARNING, ERROR, CRITICAL
            logging.getLogger().setLevel(logging.DEBUG)
            if args.verbose:
                logging_format = "%(asctime)s,%(msecs)03d %(levelname)7s %(message)s, file %(filename)s, line %(lineno)d"
            else:
                logging_format = "%(asctime)s,%(msecs)03d %(levelname)7s %(message)s"
            logging.basicConfig(
                format=logging_format, datefmt="%Y-%m-%d:%H:%M:%S",
            )
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.basicConfig(format="[%(levelname)s] %(message)s")
        for yaml_cfg in args.yaml_configs:
            print("#" * 80)
            print("Executing: ", yaml_cfg)
            ex_test = job_executor.job_executor(yaml_cfg)
            ex_test.execute_jobs()
        print("#" * 80)
        print("Done!")
        print("#" * 80)

