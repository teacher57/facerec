"""Face recognition from the command line. Installed as `facerec` by setup.sh.

Run `facerec help` to see all commands, or `facerec COMMAND -h` for one command's options.
"""
import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODEL_FILES = ["face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"]

OVERVIEW = """\
Face recognition for Mac and Raspberry Pi.

commands:
  enroll_face   remember a new person (from the camera or a folder of photos)
  list_faces    list the names of all remembered people
  test          try it out: show the video (-v) and/or call job.py loudly (-j)
  run           watch the camera without a window and call job.py quietly
  help          show this overview, or `facerec help COMMAND` for one command

examples:
  facerec enroll_face --from-camera --name Arnold
  facerec enroll_face --from-file ~/photos/arnold --name Arnold
  facerec list_faces
  facerec test -v -j
  facerec run --rate 2 --log

Every command has its own help: facerec COMMAND -h"""

CAMERA_HELP = "camera to use: auto (Pi Camera Module if present, else USB/built-in), usb, picam"
RATE_HELP = "face checks per second (default 5); lower it on slow devices, the video stays full speed"
LOG_HELP = ("save a new log file in logs/ for this run and print the same lines in the terminal: "
            "one 'name probability timestamp' line per face per check")


def build_parser():
    parser = argparse.ArgumentParser(prog="facerec", usage="facerec COMMAND [options]",
                                     description=OVERVIEW,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # help=SUPPRESS: the command list is already in OVERVIEW, so don't print it twice.
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", help=argparse.SUPPRESS)

    def add_command(name, summary, details, examples):
        return commands.add_parser(
            name, help=summary, description=f"{summary}.\n\n{details}",
            epilog="examples:\n" + "\n".join(f"  facerec {e}" for e in examples),
            formatter_class=argparse.RawDescriptionHelpFormatter)

    enroll = add_command(
        "enroll_face", "Remember a new person",
        "--from-camera takes 15 shots, one every 0.5 s, while exactly one face is visible\n"
        "(waits when there is no face, pauses when there are more), with a progress bar.\n"
        "--from-file uses every picture in a folder that shows exactly one face.\n"
        "The shots are averaged into one face. Enrolling an existing name replaces it.",
        ["enroll_face --from-camera --name Arnold",
         "enroll_face --from-file ~/photos/arnold --name Arnold"])
    source = enroll.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-camera", action="store_true", help="take 15 shots from the camera")
    source.add_argument("--from-file", metavar="FOLDER",
                        help="average all pictures (.jpg .jpeg .png .bmp .webp) in FOLDER")
    enroll.add_argument("--name", help="person's name (asked in the terminal if left out)")
    enroll.add_argument("--camera", choices=["auto", "usb", "picam"], default="auto",
                        help=CAMERA_HELP)

    add_command("list_faces", "List the names of all remembered people",
                "Reads known_faces.npz only; does not open the camera or load the models.",
                ["list_faces"])

    test = add_command(
        "test", "Try it out and see what's happening",
        "At least one of -v or -j is needed.",
        ["test -v", "test -j", "test -v -j --rate 2 --log"])
    test.add_argument("-v", "--video", action="store_true",
                      help="show the camera in a window with boxes, scores, and names")
    test.add_argument("-j", "--job", action="store_true",
                      help="call job.py and print a timestamp and the faces for every call")

    run = add_command(
        "run", "Watch the camera without a window and call job.py",
        "The quiet mode for everyday use (e.g. a Raspberry Pi running all day).\n"
        "job(faces) runs at most once every SLEEP_TIME_SEC seconds (set in job.py).",
        ["run", "run --rate 2 --log --camera picam"])

    for command in (test, run):
        command.add_argument("--rate", type=float, default=5, help=RATE_HELP)
        command.add_argument("--log", action="store_true", help=LOG_HELP)
        command.add_argument("--camera", choices=["auto", "usb", "picam"], default="auto",
                             help=CAMERA_HELP)

    help_command = add_command("help", "Show all commands, or the options of one command",
                               "Same as `facerec -h`, or `facerec COMMAND -h`.",
                               ["help", "help run"])
    help_command.add_argument("topic", nargs="?", metavar="COMMAND",
                              choices=["enroll_face", "list_faces", "test", "run", "help"],
                              help="command to show the options of")
    return parser, commands


def list_faces():
    from face_recognizer import load_known_faces
    names = sorted(load_known_faces())
    if not names:
        print("No known people yet. Run: facerec enroll_face --from-camera --name NAME")
        return
    print(f"{len(names)} known {'person' if len(names) == 1 else 'people'}:")
    for name in names:
        print(name)


def main():
    parser, commands = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "help":
        topic = getattr(args, "topic", None)
        (commands.choices[topic] if topic else parser).print_help()
        return
    if args.command == "list_faces":
        list_faces()
        return

    missing = [f for f in MODEL_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        print(f"Missing model files in models/: {', '.join(missing)}. Run setup.sh first.")
        sys.exit(1)

    # Imported only now, because loading them loads the models.
    from camera import CameraError, open_camera
    from enroll import enroll_from_camera, enroll_from_folder

    if args.command in ("test", "run") and args.rate <= 0:
        print("--rate must be greater than 0.")
        sys.exit(1)
    if args.command == "test" and not (args.video or args.job):
        print("test needs -v (show video) and/or -j (call job.py), e.g. facerec test -v -j")
        sys.exit(1)

    try:
        if args.command == "enroll_face":
            name = args.name or input("Name of the person to remember: ").strip()
            if not name:
                print("No name given.")
                sys.exit(1)
            if args.from_file:
                saved = enroll_from_folder(name, args.from_file)
            else:
                saved = enroll_from_camera(name, open_camera(args.camera))
            sys.exit(0 if saved else 1)

        import job
        from watch import watch
        if args.command == "test":
            watch(open_camera(args.camera), job, rate=args.rate, log=args.log,
                  display=args.video, run_job=args.job, print_jobs=True)
        else:
            watch(open_camera(args.camera), job, rate=args.rate, log=args.log)
    except CameraError as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
