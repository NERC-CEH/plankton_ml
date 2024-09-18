# Data Version Control 

We're trying DVC (Data Version Control) in this project, for versioning data and ML models.

There's little here on the DVC side as yet - links and notes in the README about following the approach being used here for LLM testing and fine-tuning, and how we might set it up to manage the collection "externally" (keeping the data on s3 and the metadata in source control).

Other ecologies like [RO-crate](https://www.researchobject.org/ro-crate/) and [outpack](https://github.com/mrc-ide/outpack_server) share a lot of the same aims, but are more focused on research data and with possibly more community connections. For ML pipeline projects though, DVC is mature.

## Walkthrough

Following the [DVC Getting Started](https://github.com/iterative/dvc.org/blob/main/content/docs/start/index.md) 

```
dvc init
git add .dvc/config
git add .dvc/.gitignore
```

Add our JASMIN object store as a DVC remote. Use an existing bucket for simplicity; this limits us to a single collection, but that's already the case. 

For non-AWS stores, `endpointurl` is needed - set this to the same object store url defined in `.env` as `AWS_URL_ENDPOINT`

```
dvc remote add -d jasmin s3://untagged-images-lana
dvc remote modify jasmin endpointurl https://fw-plankton-o.s3-ext.jc.rl.ac.uk
```

Add access key / secret key pair as documented in the llm_eval project. These values are also defined in `.env`, and the `--local` switch prevents accidentally committing them to git:

```
dvc remote modify --local jasmin access_key_id [our access key]
dvc remote modify --local jasmin secret_access_key [our secret key]
```

Test that it works:

`dvc push`

### Add data

Our images are already in the same bucket as this `dvc` remote. Try [import-url](https://dvc.org/doc/command-reference/import-url) to add an image in remote storage to version control. TODO - ask someone with permissions to add a dedicated bucket.

There are two ways of doing this:

`dvc import-url` - supports a `--no-download` option, creates a .dvc tracking file per object which can be added to our git repo.
`dvc stage add` - writes the tracking information into `dvc.yml`, but assumes we're downloading it and uploading to the remote.

We could also use the `--to-remote` option to transfer the data to remote storage in JASMIN's object store. We already have copies of the data in s3.

`dvc add / dvc push` would be the pattern to use where we have data in a filesystem (in a JASMIN Group Workspace, or locally) and want to store and track canonical copies of it in an object store, and reuse those in experiment pipeline stages.

#### Script

Bash script provided to automate this - read all the filenames from the CSV that intake was using as a catalog, use `dvc import-url` to create tracking information for them in a `data` directory, and then commit all the tracking information to this git repository.

`bash scripts/intake_to_dvc.sh`
