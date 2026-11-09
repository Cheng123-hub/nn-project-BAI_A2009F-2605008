"""Small shared CLI; dedicated Docker images supply their own default command.

Central Command Line Interface (CLI) dispatcher for the emotion classification project.
Parses user arguments and routes tasks to training, testing, prediction, interactive demo,
or visualization report generation modules.
"""

import argparse
import json
import os
import sys


# Parse command-line arguments and dispatch to the corresponding module workflow
def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    
    # Check if a default command was specified in environment variables (e.g. for Docker entrypoints)
    default = os.environ.get("NNB_DEFAULT_COMMAND")
    if default and (not args or args[0].startswith("-")):
        args.insert(0, default)
        
    # Configure main CLI argument parser
    parser = argparse.ArgumentParser(description="NNB emotion classification")
    commands = parser.add_subparsers(dest="command")
    
    # Subcommands: 'train-final' (train Experiment 4 only) and 'train-all' (train all 5 experiments)
    for mode in ("train-final", "train-all"):
        training = commands.add_parser(
            mode,
            help=(
                "Train Experiment 4 only"
                if mode == "train-final"
                else "Train five controlled experiments"
            ),
        )
        training.add_argument(
            "--overwrite",
            action="store_true",
            help="Explicitly replace this command's saved model artifacts; historical reports are retained",
        )
        
    # Subcommand: 'test' (evaluate on test set or predict single sentence)
    testing = commands.add_parser(
        "test", help="Evaluate final model, or predict only --text"
    )
    testing.add_argument("--model", default="models/final/model.pt")
    testing.add_argument("--text")
    
    # Subcommand: 'predict' (predict one sentence with JSON output)
    prediction = commands.add_parser(
        "predict", help="Predict one new sentence without Test evaluation"
    )
    prediction.add_argument("--model", default="models/final/model.pt")
    prediction.add_argument("--text", required=True)
    
    # Subcommand: 'demo' (interactive terminal REPL loop)
    demo = commands.add_parser("demo", help="Interactive text predictions")
    demo.add_argument("--model", default="models/final/model.pt")
    
    # Subcommand: 'reports' (rebuild figures and summary tables from recorded data)
    commands.add_parser("reports", help="Rebuild saved historical figures only")
    
    options = parser.parse_args(args)
    
    try:
        # Route execution based on parsed subcommand
        if options.command in ("train-final", "train-all"):
            from .experiment import train_models

            # Dispatch training run
            train_models(options.command, options.overwrite)
        elif options.command in ("predict", "test"):
            # If text is provided, execute single-sample prediction
            if options.text is not None:
                from .predict import predict, resolve_model

                path = resolve_model(options.model)
                print(f"Model: {path}")
                print(
                    json.dumps(
                        predict(options.text, path), ensure_ascii=False, indent=2
                    )
                )
            else:
                # Otherwise, execute full test split evaluation
                from .test import evaluate_test

                evaluate_test(options.model)
        elif options.command == "demo":
            from .predict import demo

            # Launch interactive demo loop
            demo(options.model)
        elif options.command == "reports":
            from .plots import generate_reports

            # Rebuild summary figures and comparison tables
            generate_reports()
        else:
            parser.print_help()
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
