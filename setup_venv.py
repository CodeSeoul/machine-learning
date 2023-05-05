import argparse
import logging
import subprocess
import typing as t

from importlib import import_module
from pathlib import Path


def get_args() -> argparse.Namespace:
    """Get arguments required to run program

    Returns:
        argparse.Namespace: _description_
    """
    parser = argparse.ArgumentParser("Program for setting up virtual environment and installing " 
                                     "pip requirements on unix-based applications")
    parser.add_argument("-p", "--project_path",
                        required=True, 
                        type=str, 
                        help="The path of target project. E.g. 230506-intro-to-mlops")
    return parser.parse_args()


def run_python_script(python_command: t.Union[t.List, t.Tuple]) -> None:
    """
    Attempt to run python script in a shell process. 
    If it fails, fallback to "python3"

    Args:
        python_command (t.Union[t.List, t.Tuple]): The script to run and also argument. 
        E.g.
        run_python_script('venv', 'test_venv') -> python venv test_venv
        run_python_script('script.py', '-p', '10000') -> python script.py -p 10000

    Raises:
        TypeError: If the input is not valid
    """
    logger = logging.getLogger(__name__)
    
    # Sanity checks
    if not isinstance(python_command, (t.List, t.Tuple)):
        raise TypeError("python_command must be a Tuple or List. "
                        f"Passed in: {type(python_command)}")
    if not all(isinstance(item, str) for item in python_command):
        raise TypeError(f"every element passed must be a string. Passed in: {python_command}")

    try:
        subprocess.check_output(["python", *python_command])
    # FileNotFoundError triggered if "python" does not exist
    except FileNotFoundError:
        logger.error(f"Failed to run command: 'python {' '.join(python_command)}'. "
                     "Retrying using 'python3'")
        subprocess.check_output(["python3", *python_command])


def attempt_pip_install_if_not_installed(module: str) -> None:
    """
    Do pip install if the target module is not installed. 
    If the module is invalid or cannot be installed, will raise an error

    Args:
        module (str): The target module to install. 
        E.g. attempt_pip_install_if_not_installed("numpy")
    """
    logger = logging.getLogger(__name__)
    try:
        import_module(module)
    except ModuleNotFoundError:
        logger.error(f"Module: '{module}' not found. "
                     f"Attempting to install via 'pip install {module}'")
        
        try:
            subprocess.check_output(['pip', 'install', module])
        except subprocess.CalledProcessError:
            logger.error(f"Failed to install module: '{module}' via: "
                         f"'pip install {module}'. "
                         f"Trying 'pip3 install {module}'")
            subprocess.check_output(['pip3', 'install', module])
        # If we still get a CalledProcessError, chances are that we cannot install that package
        # via pypi


def pip_install_module_if_not_installed(list_of_dependencies: t.List[str]) -> t.Callable:
    """A decorator that aims to install all required dependencies if it isn't already installed.
    Will raise an error if that module is uninstallable, since it might be human error. 
    E.g. "nump" instead of "numpy"

    Args:
        list_of_dependencies (t.List[str]): The list of dependencies to install

    Returns:
        t.Callable: The decorated function
    """
    def decorator(fn: t.Callable) -> t.Callable:
        def inner(*args, **kwargs) -> t.Any:
            # Check dependencies and install if they don't exist
            for module in list_of_dependencies:
                if not isinstance(module, str):
                    raise TypeError(f'Module: {module} must be a <str>')
                attempt_pip_install_if_not_installed(module)
                
            return fn(*args, **kwargs)
        return inner
    return decorator


@pip_install_module_if_not_installed(["virtualenv"])
def create_venv(venv_path: str):
    """Given a virtual environment path, create a venv. 
    This has only been tested on my MAC so not sure if it works 
    in windows

    Args:
        venv_path (str): The path of the venv
    """
    # Create virtual environment
    run_python_script(("-m", "venv", venv_path))


def main(project_path: str) -> None:
    project_path: Path = Path(project_path)
    if not project_path.exists() and not project_path.is_dir():
        raise FileNotFoundError(f"No such directory: '{project_path}'.")

    # 1. Create venv
    venv_folder_name = str(project_path / "venv")
    create_venv(venv_path=venv_folder_name)

    # Location of target venv
    venv_python_path = f"{venv_folder_name}/bin/python"

    # 2. Update venv pip
    subprocess.run([venv_python_path, "-m", "pip", "install", "--upgrade", "pip"])

    # 3. Install dependencies
    subprocess.run([venv_python_path, "-m", "pip", "install", "-r", f"{project_path}/requirements.txt"])


if __name__ == "__main__":
    args = get_args()
    main(args.project_path)
