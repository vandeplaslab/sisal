git :
	git add .
	git commit -m "$m"
	git push origin main	


untrack:
	git rm -r --cached .
	git add .
	git commit -m ".gitignore fix"