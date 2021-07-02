from algorithms.svea import SVEA

algorithm = {
	'svea': SVEA
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
