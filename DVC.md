# Data Version Control 

We tried out [DVC (Data Version Control)](https://dvc.org/) in this project, for versioning data and ML models.

* Manage image collections as "external" sources (keeping the data on s3 and the metadata held in a git repository).
* Create simple reproducible pipelines for processing data, training and fine-tuning models
* Potential integration with [CML](https://cml.dev/doc/cml-with-dvc) for "continuous machine learning" - see the [llm-eval](https://github.com/NERC-CEH/llm-eval) project for a properly developed take on this.

Other ecologies like [RO-crate](https://www.researchobject.org/ro-crate/) and [outpack](https://github.com/mrc-ide/outpack_server) share some of the same aims as DVC, but are more focused on research data and with possibly more community connections. For ML pipeline projects though, DVC is mature.

## Summary

Our data transfer to s3 storage is being [managed via an API](PIPELINES.md) and we don't have frequent changes to the source data. Keeping the `dvc.lock` projects in git and using `dvc` to synchronise training data download between development machines and hosts in JASMIN is a good pattern for other projects, but not for us here.

The data pipeline included here is minimal (just a chain of scripts!). We wanted to show several different image collections and resulting models trained on their embeddings. `dvc repro` wants to destroy and recreate directories used as input/output between stages, so those have been commented out of the [example dvc.yaml](scripts/dvc.yaml). 

For publishing an experiment, reproducible as a pipeline with a couple of commands and with _little to no adaptation of existing code_ needed to get it to work, it's a decent fit.

## Walkthrough

### Setting up a "DVC remote" in object storage

Following the [DVC Getting Started](https://github.com/iterative/dvc.org/blob/main/content/docs/start/index.md) 

```
dvc init
git add .dvc/config
git add .dvc/.gitignore
```

Add our JASMIN object store as a DVC remote. Use an existing bucket for simplicity; this limits us to a single collection, but that's already the case. 

For non-AWS stores, `endpointurl` is needed - set this to the same object store url defined in `.env` as `AWS_URL_ENDPOINT`

```
dvc remote add -d jasmin s3://metadata
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

#### Example for `dvc add`

Files are on our local system. They might already be version controlled in a git repository!

This uploads a image file to our object storage, and creates a .dvc metadata file in our local directory, alongside the image:

```
dvc add tests/fixtures/test_images/testymctestface_113.tif --to-remote`                                                                       
git add tests/fixtures/test_images/testymctestface_113.tif.dvc
git commit -m "add dvc metadata"
git push origin our_branch
```

Now in a completely separate checkout of the git repository, with the dvc remote set up to point to the same storage, we can

`dvc pull`

And this downloads the copy of the image linked to the metadata. It's quite nice!

#### Script to `import-url`

Bash script provided to automate this - read all the filenames from the CSV that intake was using as a catalog, use `dvc import-url` to create tracking information for them in a `data` directory, and then commit all the tracking information to this git repository.

`bash scripts/intake_to_dvc.sh`

### Define pipeline stages

We want a [dvc.yaml](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files) to keep our pipeline definition(s) in.

Each script is converted to a stage; pass the output of one as the input of the next as a directory path (the `-d` switch).

Option of a `params.yaml` with the `-p` switch which stores hyperparameters / initialisation values per stage. 

Use `dvc` to chain the existing scripts together into a pipeline:

`cd scripts` - write a `dvc.yaml` into this directory.

Rebuild the index of images in our s3 store:

`dvc stage add -n index python image_metadata.py`

Use that index to extract and store embeddings from images:

`dvc stage add -n embeddings -o ../vectors python image_embeddings.py`

Then check we can run our two-stage pipeline:

`dvc repro`

This creates a `dvc.lock` to commit to the repository, and suggests a `dvc push` which sends some amount of experiment metadata to the remote.

When we run `dvc repro` again the second stage detects no change and doesn't re-run; but as our first stage only wrote a file back to `s3`, not to the filesystem, it may not be the behaviour we want.

Now its output path `../vectors` is available to use as input to a model-building stage.

Add a script that fits a K-means model from the image embeddings and saves it (hoping it saves automatically into `../models`)

`dvc stage add -n cluster -d ../vectors -o ../models cluster.py`

`dvc repro` at this point does want to run the image embeddings again.


## References

* [DVC with s3](https://github.com/NERC-CEH/llm-eval/blob/main/dvc.md) condensed walkthrough as part of the LLM evaluation project - complete this up to `dvc remote modify...` to set up the s3 connection.

* [Tutorial: versioning data and models: What's next?](https://dvc.org/doc/use-cases/versioning-data-and-models/tutorial#whats-next) 

* [Importing external data: Avoiding duplication](https://dvc.org/doc/user-guide/data-management/importing-external-data#avoiding-duplication) - is it this pattern?



