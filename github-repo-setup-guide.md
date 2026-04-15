# GitHub Portfolio Repository Setup Guide

## Complete Instructions for Creating Your AI/ML Technical Writing Portfolio

---

## Prerequisites

Make sure you have:
- A GitHub account (create one at github.com if needed)
- Git installed on your computer
- Basic familiarity with terminal/command line

---

## Step 1: Create the Repository on GitHub

### Option A: Create Empty Repository First

1. Go to **github.com** and click **+** → **New repository**
2. Repository name: `ai-ml-technical-writing-portfolio` (or your preference)
3. Description: "AI/ML Technical Writing Portfolio — API docs, tutorials, technical articles"
4. Visibility: **Public** (recruiters need to see it!)
5. **Do NOT** initialize with README (we have our own)
6. Click **Create repository**

### Option B: Create via GitHub CLI (if installed)

```bash
gh repo create ai-ml-technical-writing-portfolio \
  --public \
  --description "AI/ML Technical Writing Portfolio — API docs, tutorials, technical articles"
```

---

## Step 2: Clone the Repository

```bash
# Navigate to your projects folder
cd ~/projects  # or wherever you keep your work

# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-ml-technical-writing-portfolio.git

cd ai-ml-technical-writing-portfolio
```

---

## Step 3: Copy Portfolio Files

Assuming your portfolio files are in `./portfolio/`:

```bash
# From the directory containing your portfolio folder
cp ./portfolio/*.md ~/projects/ai-ml-technical-writing-portfolio/
cp ./portfolio/.gitignore ~/projects/ai-ml-technical-writing-portfolio/
```

Or if you're already in the portfolio folder:

```bash
cd /path/to/portfolio

# Copy all markdown files and gitignore to the repo
cp *.md ~/projects/ai-ml-technical-writing-portfolio/
cp .gitignore ~/projects/ai-ml-technical-writing-portfolio/
```

---

## Step 4: Verify Files Copied

```bash
cd ~/projects/ai-ml-technical-writing-portfolio
ls -la
```

You should see:
```
README.md
CONTRIBUTING.md
.gitignore
documind-api-docs.md
rag-tutorial-part1.md
rag-tutorial-part2.md
rag-tutorial-part3.md
how-attention-works.md
linkedin-profile-guide.md
github-repo-setup-guide.md
```

---

## Step 5: Initialize Git and Commit

```bash
# Navigate to repository
cd ~/projects/ai-ml-technical-writing-portfolio

# Check status (should show untracked files)
git status

# Add all files
git add .

# Verify what will be committed
git status

# Make initial commit
git commit -m "Initial portfolio: API docs, RAG tutorial series, attention article"
```

---

## Step 6: Push to GitHub

```bash
# Set upstream and push
git push -u origin main

# If your default branch is 'master' instead of 'main':
git push -u origin master
```

---

## Step 7: Verify on GitHub

Visit your repository:
```
https://github.com/YOUR_USERNAME/ai-ml-technical-writing-portfolio
```

Verify:
- ✅ README.md renders properly as the main page
- ✅ All markdown files are visible
- ✅ Code syntax highlighting works in tutorials
- ✅ Links work (if any)

---

## Step 8: Configure Repository Settings

### A. Add Topics/Tags

On GitHub, go to **Settings** → **General** → **Topics**

Add these topics:
```
technical-writing    ai-machine-learning    documentation
api-documentation   tutorials               portfolio
llm                 rag-systems             transformers
```

### B. Set Default Branch Name (Optional)

If you want `main` instead of `master`:
- **Settings** → **Branches** → Change default branch to `main`

### C. Enable GitHub Pages (for nice URL)

**Settings** → **Pages** → **Source**: `main` branch, `/ (root)` folder

Your portfolio will be available at:
```
https://YOUR_USERNAME.github.io/ai-ml-technical-writing-portfolio/
```

---

## Step 9: Pin the Repository to Your Profile

1. Go to your GitHub profile page
2. Click **Edit profile**
3. Scroll to **Pinned repositories** section
4. Click **+** next to "Pin a repository"
5. Select `ai-ml-technical-writing-portfolio`
6. Click **Confirm selection**
7. Click **Save changes**

This ensures visitors see your portfolio first!

---

## Step 10: Add Portfolio Link to Other Profiles

### LinkedIn:
- Edit profile → "Featured" section → Add link
- Or add to "About" section

### Resume:
Add to header or contacts section:
```
Portfolio: github.com/YOUR_USERNAME/ai-ml-technical-writing-portfolio
```

---

## Troubleshooting

### Error: "Permission denied (publickey)"

You need to set up SSH keys OR use HTTPS URL:

```bash
# Change remote to HTTPS
git remote set-url origin https://github.com/YOUR_USERNAME/ai-ml-technical-writing-portfolio.git
```

### Error: "Repository not found"

Double-check:
1. Repository was created on GitHub
2. Username is correct in URL
3. You have internet connection

### Markdown Not Rendering Properly

GitHub renders markdown automatically. If something looks wrong:
- Check for proper heading syntax (`#`, `##`, etc.)
- Verify code blocks use triple backticks (```) 
- Tables need proper pipe (|) alignment

---

## Optional: Add a Custom Domain

If you want `portfolio.yourname.com` instead of GitHub Pages URL:

1. Buy domain from registrar (Namecheap, GoDaddy, etc.)
2. Configure DNS to point to GitHub Pages
3. **Settings** → **Pages** → Add custom domain

---

## Optional: Add Analytics

Track who visits your portfolio:

### Using Simple JavaScript Counter:

Add to README.md:
```html
<!-- In README, anywhere -->
<img src="https://profile-counter.glitch.me/YOUR_USERNAME/count.svg" alt="visitor count">
```

### Using Google Analytics (more detailed):

1. Create GA4 property at analytics.google.com
2. Get measurement ID (G-XXXXXXXXXX)
3. Add gtag.js to a custom HTML file
4. Include on GitHub Pages

---

## Final Checklist

Before sharing your portfolio:

- [ ] Repository is public
- [ ] README renders correctly
- [ ] All markdown files display properly
- [ ] Code blocks have syntax highlighting
- [ ] ASCII diagrams render correctly
- [ ] No broken links
- [ ] Repository is pinned to profile
- [ ] Portfolio link added to LinkedIn/resume
- [ ] Tested on mobile view
- [ ] Asked a friend to review it

---

## Sample Portfolio URL Format

Once complete, your portfolio URLs will be:

**Repository:** `https://github.com/YOUR_USERNAME/ai-ml-technical-writing-portfolio`

**GitHub Pages:** `https://YOUR_USERNAME.github.io/ai-ml-technical-writing-portfolio/`

**Individual pieces:**
- API Docs: `/documind-api-docs.md`
- RAG Tutorial Part 1: `/rag-tutorial-part1.md`
- How Attention Works: `/how-attention-works.md`

---

## Next Steps After Setup

1. **Share with network** — Post on LinkedIn, tell friends
2. **Add to job applications** — Include link in every application
3. **Monitor analytics** — See what people read most
4. **Update regularly** — Add new pieces as you create them
5. **Gather feedback** — Ask readers what they think

---

*Good luck with your portfolio! 🚀*
