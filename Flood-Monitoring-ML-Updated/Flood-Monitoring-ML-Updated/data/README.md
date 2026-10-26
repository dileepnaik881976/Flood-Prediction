# data/

Place the four CSV files here (they are NOT included in the repository or ZIP):

    train.csv, test.csv, sample_submission.csv, flood.csv

They are needed for training (`python -m src.train`), evaluation (`python -m src.evaluate`), tests that
load the datasets, and generating `submissions/submission.csv`. The dashboard itself does NOT need them: it
uses the saved model and the precomputed files in `reports/`.

Keep large, private or licensed datasets out of public repositories (see .gitignore). If you must version
them, use Git LFS or share a download link in the README instead.
