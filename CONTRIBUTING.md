# Contributing

Thank you for your interest in contributing to RAG Master Class!

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

## Development Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-master-class.git
cd rag-master-class

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r Classical-RAG/requirements.txt
pip install -r Agentic-RAG/requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Guidelines

- Write clear commit messages
- Add docstrings to new functions and classes
- Keep code consistent with the existing style
- Test your changes before submitting a PR
- Update README if you add new features

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bugs
- Provide your Python version and OS

## Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Keep functions focused and small

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
