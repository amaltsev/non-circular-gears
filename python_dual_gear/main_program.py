from gear_tooth import add_teeth
from optimization.optimization import optimize_center
from report import Reporter, ReportingSuite
from drive_gears.models import our_models, Model, find_model_by_name, retrieve_model_from_folder, retrieve_models_from_folder
from drive_gears.shape_processor import *
from core.compute_dual_gear import compute_dual_gear
from core.rotate_and_carve import rotate_and_carve
import fabrication
import fabsvg
import drive_gears.shape_factory as shape_factory
import logging
import sys
from plot.plot_sampled_function import rotate
import yaml
from plot.qt_plot import Plotter
import os
from typing import Optional, Iterable, List, Tuple
from core.dual_optimization import align_and_average, contour_distance, rebuild_polar
from util_functions import save_contour
import matplotlib.pyplot as plt
from time import perf_counter_ns
import textwrap


# writing log to file
if not os.path.exists(Reporter.pre_fix):
    os.mkdir(Reporter.pre_fix)
logging.basicConfig(filename=os.path.join('debug','info.log'), level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger = logging.getLogger(__name__)


def get_inputs(debugger, drive_model, driven_model, plotter, uniform=True):
    cart_drive = shape_factory.get_shape_contour(drive_model, uniform=uniform, plots=None)
    cart_driven = shape_factory.get_shape_contour(driven_model, uniform=uniform, plots=None)
    if plotter is not None:
        plotter.draw_contours(debugger.file_path('input_drive.png'), [('input_drive', cart_drive)], None)
        plotter.draw_contours(debugger.file_path('input_driven.png'), [('input_driven', cart_driven)], None)
    logging.debug('original 3D meshes generated')
    return cart_drive, cart_driven


def init_debugger(models: Iterable[Model], opt_config):
    debugger = Reporter([model.name for model in models])

    logging_fh = logging.FileHandler(debugger.file_path('logs.log'), 'w')
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))

    logging.getLogger('').addHandler(logging_fh)

    return debugger


def main(opt_config_file='optimization_config.yaml',
         task_config_file=None,
         drive_model_code=None,
         driven_model_code=None):

    # Defaults in the optimization config
    #
    if opt_config_file is None:
        raise RuntimeError('A configuration file is required')

    if os.path.split(opt_config_file)[0] == '':
        opt_config_file = os.path.join(os.path.dirname(__file__), opt_config_file)

    with open(opt_config_file) as opt_config_fd:
        opt_config = yaml.safe_load(opt_config_fd)

    ### print(opt_config)

    logger.info(f"Defaults from '{opt_config_file}'")

    # Overriding with task config
    #
    if task_config_file is not None:
        with open(task_config_file) as task_config_fd:
            task_config = yaml.safe_load(task_config_fd)
            opt_config.update(task_config)
            logger.info(f"Task overrides from '{task_config_file}'")

    opt_config['sampling_count'] = tuple(opt_config['sampling_count'])

    ### print(opt_config)

    logging.debug(f'Optimization read from {opt_config_file}:' + repr(opt_config))

    # What are the models? Typically listed in the task config
    #
    if not drive_model_code:
        drive_model_code = opt_config.get('drive_model')
        if not drive_model_code:
            raise RuntimeError("No 'drive_model' provided in task or command line")

    if not driven_model_code:
        driven_model_code = opt_config.get('driven_model')
        if not driven_model_code:
            raise RuntimeError("No 'drive_model' provided in task or command line")

    logger.info(f"Drive model '{drive_model_code}', driven model '{driven_model_code}'")

    drive_model = find_model_by_name(drive_model_code)
    driven_model = find_model_by_name(driven_model_code)

    # Debugger path depends on models
    #
    debugger = init_debugger((drive_model, driven_model), opt_config)

    # Model parameters are usually overridden in the task config
    #
    if float(opt_config.get('tooth_height', 0)) > 0:
        drive_model.tooth_height = float(opt_config.get('tooth_height', 0))
        driven_model.tooth_height = float(opt_config.get('tooth_height', 0))
        logger.info(f'Tooth height: {drive_model.tooth_height}')

    if int(opt_config.get('tooth_num', 0)) > 0:
        drive_model.tooth_num = int(opt_config.get('tooth_num', 0))
        drive_model.sample_num = drive_model.tooth_num * drive_model.oversampling
        driven_model.tooth_num = int(opt_config.get('tooth_num', 0))
        driven_model.sample_num = driven_model.tooth_num * driven_model.oversampling
        logger.info(f'Tooth number: {drive_model.tooth_num}')

    # Gear transfer ratio, typically 1:1
    #
    gear_ratio = opt_config.get('gear_ratio') or opt_config.get('k') or 1
    opt_config['k'] = gear_ratio

    # Processing
    #
    logger.info(f'Optimizing {drive_model.name} with {driven_model.name}')

    model_names = f'{drive_model.name}, {driven_model.name}'

    plotter = Plotter()

    plt.close('all')

    # get input polygons
    #
    cart_input_drive, cart_input_driven = get_inputs(debugger, drive_model, driven_model, plotter, uniform=True)

    logger.info(f"...pre-processing done for '{model_names}'")

    # optimization
    #
    center, center_distance, cart_drive, score = optimize_center(cart_input_drive,
                                                                 cart_input_driven,
                                                                 debugger,
                                                                 opt_config,
                                                                 plotter,
                                                                 k=gear_ratio)
    print('optimization done for ' + model_names)
    logger.info(f'score = {score}')

    drive_model.center_point = (0, 0)
    cart_drive = shape_factory.uniform_and_smooth(cart_drive, drive_model)

    *_, center_distance, phi = compute_dual_gear(toExteriorPolarCoord(Point(0, 0), cart_drive, drive_model.tooth_num * 32), gear_ratio)

    # add teeth
    cart_drive_gear = add_teeth((0, 0), center_distance, debugger, cart_drive, drive_model, plotter)

    # rotate and cut
    cart_driven_gear = rotate_and_carve(cart_drive_gear, (0, 0), center_distance, debugger, drive_model, phi, None, k=gear_ratio,
                                        replay_anim=False, save_anim=False)

    ### logger.info(f'Driven gear mesh angle: {phi[0]}')

    # Done, saving the output
    # TODO: Convert to a fabricator class?
    #
    output_type = opt_config.get('output_type') or 'svg'
    output_file = opt_config.get('output_file')

    match output_type:

        # save 2D SVG of both shapes
        case 'svg':
            svgfile = fabsvg.generate_2d_svg(debugger, output_file,
                                             cart_drive_gear, (0,0), 0,
                                             cart_driven_gear, (center_distance, 0), phi[0])
            logger.info(f'Output SVG: {svgfile}')

        # save 2D contour and 3D meshes with axle holes
        case 'obj':
            output_file = output_file or 'output.obj'
            drive_2d_file = output_file.replace('.obj', '-drive-2d-(0,0).obj')
            driven_2d_file = output_file.replace('.obj', f'-driven-2d-({center_distance},0).obj')
            drive_3d_file = output_file.replace('.obj', '-drive-3d-(0,0).obj')
            driven_3d_file = output_file.replace('.obj', f'-driven-3d-({center_distance},0).obj')

            fabrication.generate_2d_obj(debugger, drive_2d_file, cart_drive_gear)
            fabrication.generate_2d_obj(debugger, driven_2d_file, cart_driven_gear)

            fabrication.generate_3D_with_axles(8,
                drive_2d_file,
                driven_2d_file,
                (0, 0), (center_distance, 0),
                debugger, 6,
                drive_mesh_file=drive_3d_file,
                driven_mesh_file=driven_3d_file)

        case _:
            raise RuntimeError('Unsupported output_type')


### def gradual_average(drive_model: Model, driven_model: Model, drive_center: Tuple[float, float],
###                     driven_center: Tuple[float, float], count_of_averages: int, opt_config_file='optimization_config.yaml'):
###     """
###     Gradually average two contours
###     :param drive_model: The drive model
###     :param driven_model: The driven model
###     :param drive_center: center of drive
###     :param driven_center: center of driven
###     :param count_of_averages: count of average values
###     :return: None
###     """
###     debugger, opt_config, plotter = init((drive_model, driven_model), opt_config_file)
###     drive_contour, driven_contour = get_inputs(debugger, drive_model, driven_model, plotter)
###
###     distance, d_drive, d_driven, dist_drive, dist_driven = \
###         contour_distance(drive_contour, drive_center, driven_contour, driven_center, 1024)
###     for average in np.linspace(0, 1, count_of_averages, True):
###         center_dist = dist_drive * 0.5 + dist_driven * 0.5
###         reconstructed_drive = rebuild_polar(center_dist, align_and_average(d_drive, d_driven, average))
###         reconstructed_driven, center_dist, phi = compute_dual_gear(list(reconstructed_drive))
###         reconstructed_drive_contour = toCartesianCoordAsNp(reconstructed_drive, 0, 0)
###         reconstructed_driven_contour = toCartesianCoordAsNp(reconstructed_driven, center_dist, 0)
###         reconstructed_driven_contour = np.array(rotate(reconstructed_driven_contour, phi[0], (center_dist, 0)))
###         average_str = '%1.8f' % average
###         plotter.draw_contours(debugger.file_path(average_str + '.png'), [
###             ('math_drive', reconstructed_drive_contour),
###             ('math_driven', reconstructed_driven_contour)
###         ], [(0, 0), (center_dist, 0)])
###         save_contour(debugger.file_path(average_str + '_drive.dat'), reconstructed_drive_contour)
###         save_contour(debugger.file_path(average_str + '_driven.dat'), reconstructed_driven_contour)


if __name__ == '__main__':
    usage = textwrap.dedent("""\
        Usage:
          python {__file__}.py <parameters.yaml> [model1] [model2]

        Description:
          <task.yaml> : task parameters
          [model1]    : first model name (default from parameters)
          [model2]    : second model name (default from parameters)
    """)

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print(usage)
        sys.exit(1)

    task_config_file = sys.argv[1]
    model1 = sys.argv[2] if len(sys.argv) > 2 else None
    model2 = sys.argv[3] if len(sys.argv) > 3 else None

    main(task_config_file=task_config_file,
         drive_model_code = model1,
         driven_model_code = model2)