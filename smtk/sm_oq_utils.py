from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.scalerel.peer import PeerMSR
from openquake.hazardlib.gsim.gmpe_table import GMPETable
from openquake.hazardlib.gsim.base import GMPE

# Get a list of the available GSIMs
AVAILABLE_GSIMS = get_available_gsims()

# Regular expression to get a GMPETable from string:
_gmpetable_regex = re.compile(r'^GMPETable\(([^)]+?)\)$')


def check_gsim_list(gsim_list):
    """
    Check the GSIM models or strings in `gsim_list`, and return a dict of
    gsim names (str) mapped to their :class:`openquake.hazardlib.Gsim`.
    Raises error if any Gsim in the list is supported in OpenQuake.

    If a Gsim is passed as instance, its string representation is inferred
    from the class name and optional arguments. If a Gsim is passed as string,
    the associated class name is fetched from the OpenQuake available Gsims.

    :param gsim_list: list of GSIM names (str) or OpenQuake Gsims
    :return: a dict of GSIM names (str) mapped to the associated GSIM
    """
    output_gsims = {}
    for gs in gsim_list:
        if isinstance(gs, GMPE):
            output_gsims[_get_gmpe_name(gs)] = gs  # get name of GMPE instance
        elif gs in AVAILABLE_GSIMS:
            output_gsims[gs] = AVAILABLE_GSIMS[gs]()
        else:
            match = _gmpetable_regex.match(gs)  # GMPETable ?
            if match:
                filepath = match.group(1).split("=")[1]  # get table filename
                output_gsims[gs] = GMPETable(gmpe_table=filepath)
            else:
                raise ValueError('%s Not supported by OpenQuake' % gs)

    return output_gsims


def _get_gmpe_name(gsim):
    """
    Returns the name of the GMPE given an instance of the class
    """
    match = _gmpetable_regex.match(str(gsim))  # GMPETable ?
    if match:
        filepath = match.group(1).split("=")[1][1:-1]
        return 'GMPETable(gmpe_table=%s)' % filepath
    else:
        gsim_name = gsim.__class__.__name__
        additional_args = []
        # Build the GSIM string by showing name and arguments. Keep things
        # simple (no replacements, no case changes) as we might want to be able
        # to get back the GSIM from its string in the future.
        for key in gsim.__dict__:
            if key.startswith("kwargs"):
                continue
            val = str(gsim.__dict__[key])  # quoting strings with json maybe?
            additional_args.append("{:s}={:s}".format(key, val))
        if len(additional_args):
            gsim_name_str = "({:s})".format(", ".join(additional_args))
            return gsim_name + gsim_name_str
        else:
            return gsim_name


DEFAULT_MSR = PeerMSR()

# Moved from sm_database: Mechanism type to Rake conversion:
MECHANISM_TYPE = {
    "Normal": -90.0,
    "Strike-Slip": 0.0,
    "Reverse": 90.0,
    "Oblique": 0.0,
    "Unknown": 0.0,
    "N": -90.0,  # Flatfile conventions
    "S": 0.0,
    "R": 90.0,
    "U": 0.0,
    "NF": -90.,  # ESM flatfile conventions
    "SS": 0.,
    "TF": 90.,
    "NS": -45.,  # Normal with strike-slip component
    "TS": 45.,  # Reverse with strike-slip component
    "O": 0.0
}

DIP_TYPE = {
    "Normal": 60.0,
    "Strike-Slip": 90.0,
    "Reverse": 35.0,
    "Oblique": 60.0,
    "Unknown": 90.0,
    "N": 60.0,  # Flatfile conventions
    "S": 90.0,
    "R": 35.0,
    "U": 90.0,
    "NF": 60.,  # ESM flatfile conventions
    "SS": 90.,
    "TF": 35.,
    "NS": 70.,  # Normal with strike-slip component
    "TS": 45.,  # Reverse with strike-slip component
    "O": 90.0
}
