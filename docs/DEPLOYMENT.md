# 🚀 RENDER DEPLOYMENT GUIDE: Auto-Analyst Platform

## STEP-BY-STEP DEPLOYMENT INSTRUCTIONS

### 1. PREREQUISITE SETUP

**A) GitHub Repository Preparation**
```bash
# 1. Apply all fixes to your repository
git add .
git commit -m "fix: Production-ready fixes for Render deployment"
git push origin main

# 2. Verify required files exist:
- requirements.txt ✅
- .env.example ✅  
- render.yaml ✅
- runtime.txt ✅
- backend/__init__.py ✅
```

**B) Environment Variables Preparation**
Generate secure keys for production:
```bash
# Generate SECRET_KEY (32+ characters)
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY (32+ characters) 
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
```

---

### 2. RENDER SETUP PROCESS

**Step 1: Create PostgreSQL Database**
1. Log into Render Dashboard (https://dashboard.render.com)
2. Click "New" → "PostgreSQL"  
3. Configure database:
   - **Name**: `auto-analyst-db`
   - **Database Name**: `auto_analyst`
   - **User**: `auto_analyst_user`
   - **Region**: `Oregon` (US-West)
   - **Plan**: `Free` (for testing)
4. Click "Create Database"
5. **CRITICAL**: Copy the connection string when ready

**Step 2: Create Web Service**
1. Click "New" → "Web Service"
2. Connect your GitHub repository
3. Configure service:
   - **Name**: `auto-analyst-backend`
   - **Environment**: `Python`
   - **Region**: `Oregon`
   - **Branch**: `main`
   - **Build Command**: 
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300
     ```

**Step 3: Environment Variables Configuration**
Add these environment variables in Render dashboard:

| Variable | Value | Source |
|----------|--------|--------|
| `ENVIRONMENT` | `production` | Manual |
| `DEBUG` | `false` | Manual |
| `SECRET_KEY` | `[Generated 32+ char key]` | Manual |
| `JWT_SECRET_KEY` | `[Generated 32+ char key]` | Manual |
| `DATABASE_URL` | `[From PostgreSQL service]` | Database Connection String |
| `PYTHONPATH` | `/opt/render/project/src` | Manual |
| `PYTHONUNBUFFERED` | `1` | Manual |
| `LOG_LEVEL` | `INFO` | Manual |
| `CORS_ORIGINS` | `["*"]` | Manual |
| `PORT` | `8000` | Auto (Render sets this) |

---

### 3. DEPLOYMENT EXECUTION

**Step 4: Initial Deployment**
1. Click "Create Web Service"
2. Monitor build logs for errors
3. Expected build time: 5-10 minutes
4. Watch for successful completion

**Step 5: Database Migration**
After successful build, run migrations:
1. Go to "Environment" tab in your web service
2. Add a "Deploy Hook" (if available) or run manually:
   ```bash
   alembic upgrade head
   ```

**Step 6: Health Check Verification**
1. Visit: `https://your-service-name.onrender.com/health`
2. Expected response:
   ```json
   {
     "status": "healthy",
     "version": "2.0.0",
     "services": {
       "backend": true,
       "database": true
     }
   }
   ```

---

### 4. TROUBLESHOOTING COMMON ISSUES

**Issue 1: Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'backend'
# Solution: Verify PYTHONPATH environment variable
PYTHONPATH=/opt/render/project/src
```

**Issue 2: Database Connection Errors**
```bash
# Error: connection to server failed
# Solution: Check DATABASE_URL format
# Ensure: postgresql://username:password@host:port/database
```

**Issue 3: Build Timeout**
```bash
# Error: Build exceeded time limit
# Solution: Remove heavy ML dependencies temporarily:
# Comment out tensorflow, torch in requirements.txt
```

**Issue 4: Memory Issues**
```bash
# Error: Process exceeded memory limit  
# Solution: Use Free tier limitations
# Reduce worker count: -w 2 (instead of -w 4)
```

---

### 5. POST-DEPLOYMENT VERIFICATION

**Endpoint Testing**
```bash
# Health check
curl https://your-service.onrender.com/health

# API status
curl https://your-service.onrender.com/api/v1/status  

# Database connectivity
curl https://your-service.onrender.com/

# Docs (if enabled)
curl https://your-service.onrender.com/docs
```

**Performance Monitoring**
- Check Render service metrics
- Monitor response times
- Watch for error rates
- Verify database connection pool usage

---

### 6. PRODUCTION OPTIMIZATIONS

**Security Hardening**
- [ ] Rotate SECRET_KEY and JWT_SECRET_KEY regularly
- [ ] Set specific CORS_ORIGINS (remove "*")
- [ ] Enable HTTPS only
- [ ] Add rate limiting configuration
- [ ] Review and audit environment variables

**Performance Tuning**
- [ ] Monitor database query performance
- [ ] Set up Redis for caching (paid plan)
- [ ] Configure proper logging levels
- [ ] Optimize Docker image size
- [ ] Set up CDN for static files

**Monitoring & Alerting**
- [ ] Set up Render service monitoring
- [ ] Configure error tracking (Sentry)
- [ ] Add performance monitoring (New Relic)
- [ ] Set up uptime monitoring
- [ ] Configure backup strategies

---

### 7. SCALING CONSIDERATIONS

**Horizontal Scaling**
- Upgrade to paid plan for multiple instances
- Configure load balancer settings
- Implement session management with Redis
- Plan for database read replicas

**Vertical Scaling**
- Monitor resource usage patterns
- Plan memory and CPU upgrades
- Consider background task workers
- Implement caching strategies

---

## 🎯 DEPLOYMENT CHECKLIST

### Pre-Deployment
- [ ] All fixes applied from audit report
- [ ] Environment variables generated and secured
- [ ] GitHub repository updated with latest changes
- [ ] Dependencies optimized and tested

### During Deployment  
- [ ] PostgreSQL database created
- [ ] Web service configured with correct build/start commands
- [ ] Environment variables set correctly
- [ ] Build completed without errors
- [ ] Database migrations run successfully

### Post-Deployment
- [ ] Health endpoints responding correctly
- [ ] Database connectivity verified
- [ ] API endpoints functional
- [ ] Error logging working
- [ ] Performance metrics baseline established
- [ ] Security configurations validated

---

## 📞 SUPPORT & RESOURCES

**Render Documentation**: https://render.com/docs
**Auto-Analyst Issues**: https://github.com/mXrahul01/Auto-Data-Analyst/issues
**FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/

**Emergency Rollback Plan**:
1. Revert to previous GitHub commit
2. Trigger manual redeploy in Render
3. Verify health endpoints
4. Contact support if needed
