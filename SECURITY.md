# Security Policy

## Supported Versions

We release patches for security vulnerabilities. The following versions are currently supported:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@liquidui.dev** (or your designated security contact)

### What to Include

Please include the following information in your report:

1. **Type of vulnerability** (e.g., SQL injection, XSS, authentication bypass)
2. **Full paths** of source file(s) related to the vulnerability
3. **Location** of the affected source code (tag/branch/commit or direct URL)
4. **Step-by-step instructions** to reproduce the issue
5. **Proof-of-concept or exploit code** (if possible)
6. **Impact** of the vulnerability (what an attacker could achieve)
7. **Suggested fix** (if you have one)

### What to Expect

- **Acknowledgment**: We'll acknowledge receipt within 48 hours
- **Initial Assessment**: We'll provide an initial assessment within 5 business days
- **Updates**: We'll keep you informed about our progress
- **Fix Timeline**: Critical vulnerabilities will be prioritized for immediate patching
- **Credit**: With your permission, we'll credit you in the security advisory

## Security Best Practices for Users

### API Keys & Secrets

**Never commit sensitive data to version control:**

```bash
# Bad ❌
ALPACA_API_KEY = "your_actual_key_here"

# Good ✅
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
```

**Use the provided .env.example template:**

```bash
cp .env.example .env
# Edit .env with your real keys
# .env is automatically ignored by git
```

### Environment Variables

**Always use environment variables for:**
- API keys
- Database passwords
- Secret keys
- Any sensitive configuration

**Use a secrets manager in production:**
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- Google Secret Manager

### Database Security

**Never use default credentials:**

```yaml
# Bad ❌
POSTGRES_PASSWORD=changeme

# Good ✅
POSTGRES_PASSWORD=randomly_generated_strong_password_32chars
```

**Use parameterized queries:**

```python
# Bad ❌
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# Good ✅
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

### API Security

**Always validate and sanitize inputs:**

```python
from pydantic import BaseModel, validator

class OrderRequest(BaseModel):
    symbol: str
    quantity: int

    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.isalpha() or len(v) > 10:
            raise ValueError('Invalid symbol')
        return v.upper()
```

**Rate limit API endpoints:**

```python
from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/data")
@limiter.limit("100/minute")
async def get_data(request: Request):
    ...
```

### Docker Security

**Don't run containers as root:**

```dockerfile
# Create non-root user
RUN groupadd -r quant && useradd -r -g quant quant
USER quant
```

**Use specific image versions:**

```dockerfile
# Bad ❌
FROM python:latest

# Good ✅
FROM python:3.11-slim
```

**Scan images for vulnerabilities:**

```bash
docker scan liquidui:latest
```

### Dependency Security

**Keep dependencies updated:**

```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip install --upgrade -r requirements.txt
```

**Use dependency scanning in CI/CD:**

See `.github/workflows/ci.yml` for automated security checks.

## Known Security Considerations

### Financial Data Sensitivity

This software handles financial data and trading strategies. Users should:

1. **Encrypt data at rest** - Use encrypted filesystems or database encryption
2. **Encrypt data in transit** - Use TLS/SSL for all network communication
3. **Implement access controls** - Restrict who can view/modify strategies
4. **Audit all access** - Log all data access and modifications
5. **Backup securely** - Encrypt backups and store them separately

### Trading Risk

While not a traditional "security" issue, please be aware:

- **Paper trading first**: Always test strategies on paper accounts
- **Kill switches enabled**: Ensure risk management is active
- **Position limits**: Set appropriate limits before live trading
- **Monitor continuously**: Use alerting for anomalous behavior

## Compliance & Regulations

### Financial Regulations

Users are responsible for compliance with:
- SEC regulations (if trading US securities)
- MiFID II (if trading in EU)
- Local financial regulations in your jurisdiction

### Data Privacy

If handling customer data:
- Comply with GDPR (EU)
- Comply with CCPA (California)
- Implement appropriate data retention policies
- Obtain necessary consent for data processing

## Security Features

### Implemented Security Measures

- ✅ **Authentication & Authorization** - JWT-based API authentication
- ✅ **Audit Logging** - Immutable logs of all critical operations
- ✅ **Input Validation** - Pydantic models for all API inputs
- ✅ **Rate Limiting** - Prevent abuse of API endpoints
- ✅ **Secrets Management** - Environment variable-based configuration
- ✅ **Container Security** - Non-root users, minimal base images
- ✅ **Dependency Scanning** - Automated security checks in CI/CD
- ✅ **Code Analysis** - Static analysis with Bandit

### Recommendations for Production

1. **Use HTTPS only** - No unencrypted communication
2. **Implement 2FA** - For all admin accounts
3. **Network segmentation** - Isolate trading systems from public networks
4. **Regular backups** - Encrypted and tested
5. **Incident response plan** - Document procedures for security incidents
6. **Security training** - Train all users on security best practices

## Security Checklist

Before going to production:

- [ ] All secrets moved to environment variables or secrets manager
- [ ] Database credentials are strong and unique
- [ ] TLS/SSL enabled for all services
- [ ] Firewall rules configured to restrict access
- [ ] Audit logging enabled and monitored
- [ ] Backup and recovery procedures tested
- [ ] Security scanning integrated into CI/CD
- [ ] Dependencies updated and vulnerability-free
- [ ] Kill switches and risk limits tested
- [ ] Incident response plan documented
- [ ] Security review completed
- [ ] Penetration testing performed (for high-value deployments)

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Docker Benchmarks](https://www.cisecurity.org/benchmark/docker)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security.html)
- [SEC Cybersecurity Guidance](https://www.sec.gov/cybersecurity)

## Contact

For security concerns:
- **Email**: security@liquidui.dev
- **PGP Key**: [Link to PGP public key]

For general questions:
- Open an issue on GitHub (non-security related only)
- Email: support@liquidui.dev

---

**Remember**: Security is everyone's responsibility. If you see something, say something.
