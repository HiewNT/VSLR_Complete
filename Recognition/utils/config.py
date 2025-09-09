"""
Configuration constants for Recognition module
"""

# Color constants
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)

# MediaPipe hand landmarks tips
TIPS = [4, 8, 12, 16, 20]

# Character classes for recognition
CLASSES = ['A', 'B', 'C', 'D', 'DD', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'Mu', 'Munguoc', 'N', 'O', 'P', 'Q', 'R', 'Rau', 'S', 'T', 'U', 'V', 'X', 'Y']

# Character display mapping for special characters
CHAR_DISPLAY_MAP = {
    "EE": "Ê",
    "AA": "Â", 
    "OW": "Ơ",
    "AW": "Ă",
    "UW": "Ư",
    "OO": "Ô",
    "DD": "Đ"
}

# Configuration constants
IMAGE_SIZE = 300
NO_HAND_THRESHOLD = 1
TONE_FRAMES_COUNT = 30
PREDICTION_HISTORY_SIZE = 25
MIN_CONFIDENCE_THRESHOLD = 0.98
TONE_CONFIDENCE_THRESHOLD = 0.8

# Tone labels for recognition
TONE_LABELS = ['huyen', 'sac', 'hoi', 'nga', 'nang']

# Special character mappings
SPECIAL_CHARACTER_REPLACE = {
    "Mu": {"A": "Â", "O": "Ô", "E": "Ê"},
    "Munguoc": {"A": "Ă"},
    "Rau": {"O": "Ơ", "U": "Ư"}
}

SPECIAL_CHARACTER_BLOCK = {
    "Â": ["A", "Mu"], 
    "Ê": ["E", "Mu"], 
    "Ơ": ["O", "Rau"],
    "Ă": ["A", "Munguoc"], 
    "Ư": ["U", "Rau"], 
    "Ô": ["O", "Mu"]
}

VALID_BEFORE = {
    "Mu": ["A", "E", "O"], 
    "Munguoc": ["A"], 
    "Rau": ["U", "O"]
}

# Tone mapping for Vietnamese characters
TONE_MAP = {
    'A': {'sac': 'Á', 'huyen': 'À', 'hoi': 'Ả', 'nga': 'Ã', 'nang': 'Ạ'},
    'E': {'sac': 'É', 'huyen': 'È', 'hoi': 'Ẻ', 'nga': 'Ẽ', 'nang': 'Ẹ'},
    'O': {'sac': 'Ó', 'huyen': 'Ò', 'hoi': 'Ỏ', 'nga': 'Õ', 'nang': 'Ọ'},
    'I': {'sac': 'Í', 'huyen': 'Ì', 'hoi': 'Ỉ', 'nga': 'Ĩ', 'nang': 'Ị'},
    'U': {'sac': 'Ú', 'huyen': 'Ù', 'hoi': 'Ủ', 'nga': 'Ũ', 'nang': 'Ụ'},
    'Y': {'sac': 'Ý', 'huyen': 'Ỳ', 'hoi': 'Ỷ', 'nga': 'Ỹ', 'nang': 'Ỵ'},
    'Â': {'sac': 'Ấ', 'huyen': 'Ầ', 'hoi': 'Ẩ', 'nga': 'Ẫ', 'nang': 'Ậ'},
    'Ê': {'sac': 'Ế', 'huyen': 'Ề', 'hoi': 'Ể', 'nga': 'Ễ', 'nang': 'Ệ'},
    'Ô': {'sac': 'Ố', 'huyen': 'Ồ', 'hoi': 'Ổ', 'nga': 'Ỗ', 'nang': 'Ộ'},
    'Ă': {'sac': 'Ắ', 'huyen': 'Ằ', 'hoi': 'Ẳ', 'nga': 'Ẵ', 'nang': 'Ặ'},
    'Ơ': {'sac': 'Ớ', 'huyen': 'Ờ', 'hoi': 'Ở', 'nga': 'Ỡ', 'nang': 'Ợ'},
    'Ư': {'sac': 'Ứ', 'huyen': 'Ừ', 'hoi': 'Ử', 'nga': 'Ữ', 'nang': 'Ự'},
}

# Base character mapping (accented to base)
BASE_CHAR_MAP = {
    'A': 'A', 'Á': 'A', 'À': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
    'E': 'E', 'É': 'E', 'È': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
    'O': 'O', 'Ó': 'O', 'Ò': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
    'I': 'I', 'Í': 'I', 'Ì': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
    'U': 'U', 'Ú': 'U', 'Ù': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
    'Y': 'Y', 'Ý': 'Y', 'Ỳ': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
    'Â': 'Â', 'Ấ': 'Â', 'Ầ': 'Â', 'Ẩ': 'Â', 'Ẫ': 'Â', 'Ậ': 'Â',
    'Ê': 'Ê', 'Ế': 'Ê', 'Ề': 'Ê', 'Ể': 'Ê', 'Ễ': 'Ê', 'Ệ': 'Ê',
    'Ô': 'Ô', 'Ố': 'Ô', 'Ồ': 'Ô', 'Ổ': 'Ô', 'Ỗ': 'Ô', 'Ộ': 'Ô',
    'Ă': 'Ă', 'Ắ': 'Ă', 'Ằ': 'Ă', 'Ẳ': 'Ă', 'Ẵ': 'Ă', 'Ặ': 'Ă',
    'Ơ': 'Ơ', 'Ớ': 'Ơ', 'Ờ': 'Ơ', 'Ở': 'Ơ', 'Ỡ': 'Ơ', 'Ợ': 'Ơ',
    'Ư': 'Ư', 'Ứ': 'Ư', 'Ừ': 'Ư', 'Ử': 'Ư', 'Ữ': 'Ư', 'Ự': 'Ư',
    'B': 'B', 'C': 'C', 'D': 'D', 'Đ': 'Đ', 'G': 'G', 'H': 'H',
    'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'P': 'P', 'Q': 'Q',
    'R': 'R', 'S': 'S', 'T': 'T', 'V': 'V', 'X': 'X',
    'Mu': 'Mu', 'Munguoc': 'Munguoc', 'Rau': 'Rau'
}
