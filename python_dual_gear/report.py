import time
import datetime
import os
import logging
from typing import Union, Iterable, Callable, Optional
from multiprocessing import Process
import sys
from plot.qt_plot import Plotter
from matplotlib.figure import Figure

class Reporter:
    pre_fix = 'debug'       # FIXME: Sync with main_program.py usage

    def __init__(self, model_name: Union[str, Iterable[str]]):
        if isinstance(model_name, str):
            self.model_name = model_name
        else:
            self.model_name = '_'.join(model_name)

        self._debug_dir_name = os.path.join(os.path.dirname(__file__), Reporter.pre_fix,
                                            datetime.datetime.fromtimestamp(time.time()).strftime(
                                                f'%Y-%m-%d_%H-%M-%S_{self.model_name}'))

        self._debug_dir_name = os.path.join(os.path.dirname(__file__), self._debug_dir_name)

        logging.info("=================== Program Start ====================")
        logging.info(f"Output directory: {self._debug_dir_name}")

        self._init_debug_dir()

    def file_path(self, file_name):
        """
        Return file name as is when there is a path (even if './foo'), otherwise
        return a path into 'debug/'.
        """
        if os.path.split(file_name)[0] == '':
            return os.path.join(self._debug_dir_name, file_name)
        else:
            return file_name

    def get_root_debug_dir_name(self):
        return self._debug_dir_name

    def get_math_debug_dir_name(self):
        return os.path.join(self._debug_dir_name, "math_rotate")

    def get_cutting_debug_dir_name(self):
        return os.path.join(self._debug_dir_name, "cut_rotate")

    def _init_debug_dir(self):
        # init root debug dir

        if not os.path.exists(Reporter.pre_fix):
            os.mkdir(Reporter.pre_fix)
        os.mkdir(self._debug_dir_name)
        logging.info("Directory %s established" % self._debug_dir_name)
        os.mkdir(os.path.join(self._debug_dir_name, "math_rotate"))
        logging.info("Directory %s established" % os.path.join(self._debug_dir_name, "math_rotate"))
        os.mkdir(os.path.join(self._debug_dir_name, "cut_rotate"))
        logging.info("Directory %s established" % os.path.join(self._debug_dir_name, "cut_rotate"))

        stable_dir = os.path.join(os.path.abspath(os.path.join(self._debug_dir_name, '..')), 'recent')
        if os.path.islink(stable_dir):
            os.unlink(stable_dir)
        os.symlink(self._debug_dir_name, stable_dir, target_is_directory=True)
        logging.info(f"Stable link in '{stable_dir}'")


class SubprocessReporter:
    """
    the debugger that starts a subprocess for a given function
    """

    def __init__(self, debugger: Reporter, function: Callable, args: tuple = ()):
        self.debugger = debugger
        self.function = function
        self.args = args
        self.process = None

    def _func(self):
        sys.stdout = open(self.debugger.file_path('stdout.txt'), 'w')
        sys.stderr = open(self.debugger.file_path('stderr.txt'), 'w')
        self.function(*self.args)

    def start(self):
        if self.process is not None:
            raise RuntimeError("SubprocessDebugger Not Restartable")
        self.process = Process(target=self._func)
        self.process.start()

    def join(self):
        """
        wait the debugging process to finish
        """
        if self.process is None:
            raise RuntimeError("Subprocess Not Started")
        self.process.join()


class ReportingSuite:
    def __init__(self, debugger: Reporter, plotter: Optional[Plotter] = None, figure: Optional[Figure] = None,
                 path_prefix: Optional[str] = None):
        self.debugger = debugger
        self.plotter = plotter
        self.figure = figure
        self.path_prefix = path_prefix

    def sub_suite(self, additional_path_prefix):
        if self.path_prefix is None:
            return ReportingSuite(self.debugger, self.plotter, self.figure, additional_path_prefix)
        else:
            return ReportingSuite(self.debugger, self.plotter, self.figure, self.path_prefix + additional_path_prefix)
