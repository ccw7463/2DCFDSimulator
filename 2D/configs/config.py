from ml_collections import ConfigDict
Config = ConfigDict()
Config.UX = '<UX>'
Config.UY = '<UY>'
Config.DEN = '<DENSITY>'
Config.VIS = '<VISCOSITY>'
Config.MARKER = [Config.UX, Config.UY, Config.DEN, Config.VIS]
Config.RAW_PATH = 'dataset/raw/'
Config.PREPROCESSED_PATH = 'dataset/preprocessed/'