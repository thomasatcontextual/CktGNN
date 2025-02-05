import sys
import platform
import subprocess
import warnings
from importlib.util import find_spec
from pathlib import Path

# Suppress torchvision image extension warning
warnings.filterwarnings('ignore', message='Failed to load image Python extension')

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def colorize(text, color):
    """Wrap text in color codes"""
    return f"{color}{text}{RESET}"

def parse_requirements(filename='requirements.txt'):
    """Parse requirements.txt and return a list of package requirements"""
    requirements = []
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Strip version specifiers
                    package = line.split('>=')[0].split('==')[0].split('<')[0].strip()
                    requirements.append(package)
    except FileNotFoundError:
        print(colorize("Warning: requirements.txt not found", YELLOW))
        return []
    return requirements

def check_environment():
    print(f"{BOLD}System Information:{RESET}")
    print("-" * 30)
    print(f"Python version: {colorize(sys.version.split()[0], GREEN)}")
    print(f"Platform: {colorize(platform.platform(), GREEN)}")
    
    # Check environment type
    is_conda = "conda" in sys.prefix
    is_venv = sys.prefix != sys.base_prefix
    
    if is_conda:
        print(f"Environment: {colorize('Conda', GREEN)}")
        print(f"Conda prefix: {colorize(sys.prefix, GREEN)}")
        # Get conda-specific package list
        try:
            result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
            print(f"\n{BOLD}Installed conda packages:{RESET}")
            print("-" * 30)
            print(result.stdout)
        except Exception as e:
            print(colorize(f"Error getting conda package list: {e}", RED))
    elif is_venv:
        print(f"Environment: {colorize('Virtual Environment (venv)', GREEN)}")
        print(f"venv path: {colorize(sys.prefix, GREEN)}")
    else:
        print(colorize("Environment: System Python (no virtual environment)", YELLOW))
    print("-" * 30)

def check_package(package_name):
    """Check if a package is available and get its version."""
    try:
        if find_spec(package_name) is None:
            # Special case for torch_spline_conv
            if package_name == 'torch_spline_conv':
                return f"{package_name}: {colorize('INFO', YELLOW)} (optional dependency)"
            return f"{package_name}: {colorize('Not installed', RED)}"
        
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown version')
        
        # Special case for torch_spline_conv - suppress the detailed error
        if package_name == 'torch_spline_conv' and version != 'unknown version':
            return f"{package_name}: {colorize('INFO', YELLOW)} (optional dependency, version {version})"
        
        return f"{package_name}: {colorize('OK', GREEN)} ({version})"
    except Exception as e:
        # For torch_spline_conv, show a simpler message
        if package_name == 'torch_spline_conv':
            return f"{package_name}: {colorize('INFO', YELLOW)} (optional dependency)"
        return f"{package_name}: {colorize(f'Error - {str(e)}', RED)}"

def test_torch_geometric():
    """Test PyG functionality by creating a simple graph"""
    try:
        import torch
        from torch_geometric.data import Data
        
        # Create a simple graph
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[1], [2], [3]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        return True, "PyG test successful"
    except Exception as e:
        return False, f"PyG test failed: {str(e)}"

def test_torch_scatter():
    """Test torch_scatter functionality"""
    try:
        import torch
        from torch_scatter import scatter_add
        
        # Create sample data
        src = torch.randn(5)  # Changed to 1D tensor
        index = torch.tensor([0, 1, 0, 1, 2])
        out = scatter_add(src, index)
        return True, "torch_scatter test successful"
    except Exception as e:
        return False, f"torch_scatter test failed: {str(e)}"

def test_torch_sparse():
    """Test torch_sparse functionality"""
    try:
        import torch
        from torch_sparse import SparseTensor
        
        # Create a sparse tensor
        row = torch.tensor([0, 1, 1])
        col = torch.tensor([1, 0, 2])
        value = torch.tensor([1., 2., 3.])
        sparse = SparseTensor(row=row, col=col, value=value)
        return True, "torch_sparse test successful"
    except Exception as e:
        return False, f"torch_sparse test failed: {str(e)}"

def test_torch_cluster():
    """Test torch_cluster functionality"""
    try:
        import torch
        from torch_cluster import grid_cluster
        
        # Create sample positions
        pos = torch.randn(5, 2)
        cluster = grid_cluster(pos, torch.tensor([0.5, 0.5]))
        return True, "torch_cluster test successful"
    except Exception as e:
        return False, f"torch_cluster test failed: {str(e)}"

def check_imports():
    try:
        print(f"\n{BOLD}Checking required packages...{RESET}")
        print("-" * 30)
        
        # Get requirements from requirements.txt
        requirements = parse_requirements()
        
        # Core packages that should always be checked
        core_packages = {
            'torch',
            'torch_geometric',
            'numpy',
            'scipy',
            'networkx',
            'igraph',
            'gensim',
            'pandas',
            'PIL',
            'sklearn',
        }
        
        # Add package name mappings and import names
        package_aliases = {
            'pillow': 'PIL',
            'python-igraph': 'igraph',
            'torch-geometric': 'torch_geometric',
            'torch-scatter': 'torch_scatter',
            'torch-sparse': 'torch_sparse',
            'torch-cluster': 'torch_cluster',
            'torch-spline-conv': 'torch_spline_conv',
            'scikit-learn': 'sklearn',
            'scikit_learn': 'sklearn',
            'pytorch-scatter': 'torch_scatter',
            'pytorch-sparse': 'torch_sparse',
            'pytorch-cluster': 'torch_cluster',
            'pytorch-spline-conv': 'torch_spline_conv',
            'pyg': 'torch_geometric'
        }
        
        # Filter out aliases from requirements
        requirements = [package_aliases.get(pkg, pkg) for pkg in requirements]
        
        # Combine and deduplicate packages
        all_packages = sorted(set(requirements).union(core_packages))
        
        # Check all packages
        for package in all_packages:
            print(check_package(package))
        
        # Check CUDA availability if PyTorch is installed
        if find_spec('torch'):
            import torch
            print(f"\n{BOLD}PyTorch CUDA Information:{RESET}")
            print("-" * 30)
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {colorize('Yes', GREEN) if cuda_available else colorize('No', YELLOW)}")
            if cuda_available:
                print(f"CUDA version: {colorize(torch.version.cuda, GREEN)}")
                print(f"Current device: {colorize(torch.cuda.get_device_name(0), GREEN)}")
        
        # Verify PyTorch Geometric extensions
        print(f"\n{BOLD}Checking PyTorch Geometric extensions:{RESET}")
        print("-" * 30)
        for ext in ['torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv']:
            print(check_package(ext))
        
        # Add functional tests for PyG ecosystem
        print(f"\n{BOLD}Testing PyG Functionality:{RESET}")
        print("-" * 30)
        
        tests = [
            ("PyTorch Geometric", test_torch_geometric()),
            ("torch_scatter", test_torch_scatter()),
            ("torch_sparse", test_torch_sparse()),
            ("torch_cluster", test_torch_cluster()),
        ]
        
        for name, (success, message) in tests:
            status = colorize("OK", GREEN) if success else colorize("Failed", RED)
            print(f"{name}: {status} - {message}")
                
    except Exception as e:
        print(colorize(f"Error during import check: {str(e)}", RED))

def main():
    print(f"{BOLD}CktGNN Environment Validation{RESET}")
    print("=" * 30)
    check_environment()
    check_imports()
    print(colorize("\nValidation complete!", GREEN))

if __name__ == "__main__":
    main() 