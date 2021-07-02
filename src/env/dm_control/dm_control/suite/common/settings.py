import os
import numpy as np
from dm_control.suite import common
from dm_control.utils import io as resources
import xmltodict

_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_FILENAMES = [
	"./common/materials.xml",
	"./common/skybox.xml",
	"./common/visual.xml",
]


def scale_recursive(obj, field, scale):
	if isinstance(obj, dict):
		keys = obj.keys()
		if field in keys:
			obj[field] = scale * float(obj[field])
		for key in ['body', 'geom', 'joint']:
			if key in keys:
				obj[key] = scale_recursive(obj[key], field, scale)
	elif isinstance(obj, list):
		for i in range(len(obj)):
			obj[i] = scale_recursive(obj[i], field, scale)
	return obj


def get_model_and_assets_from_setting_kwargs(model_fname, setting_kwargs=None):
	""""Returns a tuple containing the model XML string and a dict of assets."""
	assets = {filename: resources.GetResource(os.path.join(_SUITE_DIR, filename))
		  for filename in _FILENAMES}

	if setting_kwargs is None:
		return common.read_model(model_fname), assets

	# Convert XML to dicts
	model = xmltodict.parse(common.read_model(model_fname))
	materials = xmltodict.parse(assets['./common/materials.xml'])
	skybox = xmltodict.parse(assets['./common/skybox.xml'])

	# Edit task-specific properties
	if 'mass' in setting_kwargs or 'damping' in setting_kwargs or 'friction' in setting_kwargs:
		if 'walker' in model_fname:
			if 'damping' in setting_kwargs:
				model['mujoco']['default']['joint']['@damping'] = setting_kwargs['damping'] * float(model['mujoco']['default']['joint']['@damping'])
			if 'friction' in setting_kwargs:
				friction = setting_kwargs['friction'] * np.array([float(s) for s in model['mujoco']['default']['geom']['@friction'].split(' ')])
				model['mujoco']['default']['geom']['@friction'] = np.array2string(friction, precision=4, separator=' ', suppress_small=True).replace('[', '').replace(']', '')
		elif 'cartpole' in model_fname:
			if 'mass' in setting_kwargs:
				model['mujoco']['worldbody']['body']['geom']['@mass'] = setting_kwargs['mass'] * float(model['mujoco']['worldbody']['body']['geom']['@mass'])
				model['mujoco']['default']['default']['geom']['@mass'] = setting_kwargs['mass'] * float(model['mujoco']['default']['default']['geom']['@mass'])
			if 'damping' in setting_kwargs:
				model['mujoco']['default']['default']['joint']['@damping'] = setting_kwargs['damping'] * float(model['mujoco']['default']['default']['joint']['@damping'])
		elif 'finger' in model_fname:
			if 'damping' in setting_kwargs:
				#model['mujoco']['default']['default']['joint']['@damping'] = setting_kwargs['damping'] * float(model['mujoco']['default']['default']['joint']['@damping'])
				model['mujoco']['worldbody']['body'][1]['joint']['@damping'] = setting_kwargs['damping'] * float(model['mujoco']['worldbody']['body'][1]['joint']['@damping'])
		elif 'ball_in_cup' in model_fname:
			if 'damping' in setting_kwargs:
				model['mujoco']['default']['default']['joint']['@damping'] = setting_kwargs['damping'] * float(model['mujoco']['default']['default']['joint']['@damping'])
		elif 'cheetah' in model_fname:
			if 'mass' in setting_kwargs:
				model['mujoco']['compiler']['@settotalmass'] = setting_kwargs['mass'] * float(model['mujoco']['compiler']['@settotalmass'])
			if 'damping' in setting_kwargs:
				model['mujoco']['worldbody'] = scale_recursive(model['mujoco']['worldbody'], '@damping', setting_kwargs['damping'])
		elif 'reacher' in model_fname:
			if 'damping' in setting_kwargs:
				model['mujoco']['default']['joint']['@damping'] = setting_kwargs['damping'] * float(model['mujoco']['default']['joint']['@damping'])
		else:
			raise ValueError('unknown model_fname: ' + model_fname)
	
	# Edit grid floor
	if 'grid_rgb1' in setting_kwargs:
		assert isinstance(setting_kwargs['grid_rgb1'], (list, tuple, np.ndarray))
		assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
		materials['mujoco']['asset']['texture']['@rgb1'] = \
			f'{setting_kwargs["grid_rgb1"][0]} {setting_kwargs["grid_rgb1"][1]} {setting_kwargs["grid_rgb1"][2]}'
	if 'grid_rgb2' in setting_kwargs:
		assert isinstance(setting_kwargs['grid_rgb2'], (list, tuple, np.ndarray))
		assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
		materials['mujoco']['asset']['texture']['@rgb2'] = \
			f'{setting_kwargs["grid_rgb2"][0]} {setting_kwargs["grid_rgb2"][1]} {setting_kwargs["grid_rgb2"][2]}'
	if 'grid_markrgb' in setting_kwargs:
		assert isinstance(setting_kwargs['grid_markrgb'], (list, tuple, np.ndarray))
		assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
		materials['mujoco']['asset']['texture']['@markrgb'] = \
			f'{setting_kwargs["grid_markrgb"][0]} {setting_kwargs["grid_markrgb"][1]} {setting_kwargs["grid_markrgb"][2]}'
	if 'grid_texrepeat' in setting_kwargs:
		assert isinstance(setting_kwargs['grid_texrepeat'], (list, tuple, np.ndarray))
		assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
		materials['mujoco']['asset']['material'][0]['@texrepeat'] = \
			f'{setting_kwargs["grid_texrepeat"][0]} {setting_kwargs["grid_texrepeat"][1]}'

	# Edit self
	if 'self_rgb' in setting_kwargs:
		assert isinstance(setting_kwargs['self_rgb'], (list, tuple, np.ndarray))
		assert materials['mujoco']['asset']['material'][1]['@name'] == 'self'
		materials['mujoco']['asset']['material'][1]['@rgba'] = \
			f'{setting_kwargs["self_rgb"][0]} {setting_kwargs["self_rgb"][1]} {setting_kwargs["self_rgb"][2]} 1'

	# Edit skybox
	if 'skybox_rgb' in setting_kwargs:
		assert isinstance(setting_kwargs['skybox_rgb'], (list, tuple, np.ndarray))
		assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
		skybox['mujoco']['asset']['texture']['@rgb1'] = \
			f'{setting_kwargs["skybox_rgb"][0]} {setting_kwargs["skybox_rgb"][1]} {setting_kwargs["skybox_rgb"][2]}'
	if 'skybox_rgb2' in setting_kwargs:
		assert isinstance(setting_kwargs['skybox_rgb2'], (list, tuple, np.ndarray))
		assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
		skybox['mujoco']['asset']['texture']['@rgb2'] = \
			f'{setting_kwargs["skybox_rgb2"][0]} {setting_kwargs["skybox_rgb2"][1]} {setting_kwargs["skybox_rgb2"][2]}'
	if 'skybox_markrgb' in setting_kwargs:
		assert isinstance(setting_kwargs['skybox_markrgb'], (list, tuple, np.ndarray))
		assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
		skybox['mujoco']['asset']['texture']['@markrgb'] = \
			f'{setting_kwargs["skybox_markrgb"][0]} {setting_kwargs["skybox_markrgb"][1]} {setting_kwargs["skybox_markrgb"][2]}'

	# Convert back to XML
	model_xml = xmltodict.unparse(model)
	assets['./common/materials.xml'] = xmltodict.unparse(materials)
	assets['./common/skybox.xml'] = xmltodict.unparse(skybox)

	return model_xml, assets
