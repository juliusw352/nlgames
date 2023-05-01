# nlgames

## A toolbox for analysing nonlocal games

### Installation
The library can be installed by running the command:

```
pip install -i https://test.pypi.org/simple/ nlgames
```

Please make sure that Numpy is already installed on your machine, as there is a bug that causes the installation process to fail if this is not the case.

All other dependencies can be found in the `pyproject.toml` file.

Alternatively, to get the latest (unstable) version, it is possible to clone the git repository. When creating a file inside the repository, use the following to import the correct file:

```
import numpy as np
from nlgames.Xorgame import Xorgame
```

For real-world use, however, the library `toqito` is likely a better fit, as it is more stable and offers a wider array of features.

### Documentation
A detailed overview of each function is available in the report.

### Bugs
Any bugs encountered can be reported at https://github.com/juliusw352/nlgames/issues.