from mitosis import Parameter

from . import sho
from . import lorenz
from . import nonlinear_pendulum
from . import cubic_sho

experiments = {"sho": sho, "lorenz": lorenz, "pendulum": nonlinear_pendulum, "sho3": cubic_sho}


def lookup_params(experiment: str, params: list):
    resolved_params = []
    experiment = experiments[experiment]
    for param in params:
        p_name, p_id = param.split("=")
        choices = experiment.__getattribute__(p_name)
        resolved_params.append(Parameter(p_id, p_name, choices[p_id]))
    return resolved_params