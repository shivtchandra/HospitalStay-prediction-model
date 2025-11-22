# Contributing to Medical Impact Predictor

Thank you for your interest in contributing to the Medical Impact Predictor! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

We welcome feature suggestions! Please:
- Check existing issues first
- Provide clear use case and rationale
- Include mockups/examples if applicable

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow code style guidelines (see below)
   - Add tests for new features
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Backend tests
   python scripts/run_api_tests.py
   
   # Frontend tests
   cd frontend && npm test
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: brief description"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style Guidelines

### Python (Backend)
- Follow [PEP 8](https://pep8.org/)
- Use type hints where appropriate
- Document functions with docstrings
- Maximum line length: 100 characters

### JavaScript/React (Frontend)
- Use ESLint configuration provided
- Prefer functional components with hooks
- Use meaningful variable names
- Add PropTypes for components

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Reference issues when applicable (#123)

## 🧪 Testing

- Write unit tests for new backend functions
- Add integration tests for API endpoints
- Test UI changes across different screen sizes
- Ensure all tests pass before submitting PR

## 📚 Documentation

- Update README.md for new features
- Add inline comments for complex logic
- Update API documentation for endpoint changes
- Include examples in docstrings

## ✅ Checklist Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No console errors or warnings
- [ ] Branch is up to date with main
- [ ] Commit messages are clear

## 🎯 Priority Areas

We're especially interested in contributions for:
- Model explainability (SHAP, LIME)
- Additional visualizations
- Performance optimizations
- Docker containerization
- CI/CD pipeline setup
- Mobile responsiveness improvements

## 📧 Questions?

Feel free to open an issue for discussion or reach out to the maintainers.

Thank you for contributing! 🙏
