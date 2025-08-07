# Contributing to VIO-SLAM

Thank you for your interest in contributing to VIO-SLAM! We welcome contributions from the community.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/vio-slam.git
   cd vio-slam
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e .[dev]
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

Run before submitting:
```bash
black src/ tests/
flake8 src/ tests/
mypy src/
pytest tests/
```

## Testing

- Write tests for new functionality
- Ensure all tests pass: `pytest tests/`
- Aim for >80% code coverage
- Use descriptive test names
- Include both unit and integration tests

## Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Test Your Changes**
   ```bash
   pytest tests/
   black --check src/ tests/
   flake8 src/ tests/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Guidelines

Use conventional commit format:
- `feat: add new loop closure algorithm`
- `fix: resolve IMU integration bug`
- `docs: update installation instructions`
- `test: add unit tests for ORB tracker`
- `refactor: improve code organization`

## Documentation

- Update README.md for major changes
- Add docstrings to all public functions
- Include type hints
- Update API documentation if needed

## Issue Reporting

When reporting bugs, please include:
- Operating system and Python version
- Steps to reproduce the issue
- Expected vs. actual behavior
- Error messages and logs
- Dataset information (if applicable)

## Feature Requests

For feature requests, please:
- Search existing issues first
- Describe the problem you're trying to solve
- Explain why this feature would be useful
- Consider implementing it yourself!

## Code Review Process

1. All PRs require at least one review
2. Address review comments promptly
3. Maintain a clean commit history
4. Ensure CI passes before merge

## Getting Help

- Check existing documentation
- Search through issues
- Ask questions in discussions
- Contact maintainers if needed

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🚀