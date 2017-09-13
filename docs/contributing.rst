.. _contributing:

:github_url: https://github.com/WIPACrepo/decotools

************
Contributing
************

If you are new to working with forks, check out `GitHub's working with forks article <https://help.github.com/articles/working-with-forks/>`_.

============================
Step 1: Creating a new issue
============================

- If you don't already have a `GitHub <http://www.github.com>`_ account, create one
- Go to the `decotools GitHub page <https://github.com/WIPACrepo/decotools>`_ and create a new issue by clicking on the "Issues" tab and then the "New issue" button

.. image:: _static/new-issue-button.png

==============================
Step 2: Forking the repository
==============================

(If you have an existing configured fork of decotools, you can skip to Step 4: Syncing an existing fork)

- From the decotools repository use the "Fork" button to fork the project into your GitHub account

.. image:: _static/fork-button.png

- This forked copy of decotools can now be cloned to your local machine using

.. code-block:: bash

    $ git clone https://github.com/<your_username>/decotools.git

=======================================
Step 3: Configuring a remote for a fork
=======================================

From your cloned copy of decotools from the previous step, list the existing remotes with

.. code-block:: bash

    $ git remote -v


You'll most likely see something like

.. code-block:: bash

    origin  https://github.com/<your_username>/decotools.git (fetch)
    origin  https://github.com/<your_username>/decotools.git (push)


To add the original decotools project repository as a remote (named "upstream") to your copy of decotools via

.. code-block:: bash

    $ git remote add upstream https://github.com/WIPACrepo/decotools.git


Now when you execute ``git remote -v``, the newly added upstream remote should be present

.. code-block:: bash

    origin  https://github.com/<your_username>/decotools.git (fetch)
    origin  https://github.com/<your_username>/decotools.git (push)
    upstream        https://github.com/WIPACrepo/decotools.git (fetch)
    upstream        https://github.com/WIPACrepo/decotools.git (push)


================================
Step 4: Syncing an existing fork
================================

To ensure that your existing fork is up-to-date with the original decotools repository, fetch the upstream commits via

.. code-block:: bash

    $ git fetch upstream


The output should look something like

.. code-block:: bash

    remote: Counting objects: xx, done.
    remote: Compressing objects: 100% (xx/xx), done.
    remote: Total xx (delta xx), reused xx (delta x)
    Unpacking objects: 100% (xx/xx), done.
    From https://github.com/WIPACrepo/decotools
     * [new branch]      master     -> upstream/master


Now the commits to the master branch of WIPACrepo/decotools are stored in your local upstream/master branch. At this point, you'll want to make sure (if you're not already) that you're on the master branch of your local repository

.. code-block:: bash

    $ git checkout master
    Switched to branch 'master'


Now you can merge the upstream/master branch into your master branch with


.. code-block:: bash

    $ git merge upstream/master


Now the master branch of your local copy of decotools should be up-to-date with the original decotools master branch!

===================================
Step 5: Create a new feature branch
===================================

Next, create a new branch for the feature you would like to develop with

.. code-block:: bash

    $ git checkout -b <new_feature_branch_name>


The output should be

.. code-block:: bash

    Switched to branch '<new_feature_branch_name>'


=========================
Step 6: Develop new code!
=========================

Now add your feature, bug fix, typo fix, etc.


=======================================
Step 7: Running tests with the new code
=======================================

Once your contribution has been added, you'll want to run the tests for this project to ensure that none of the code you added broke any tests. If you haven't already, make sure you have the necessary software installed for running the tests (pytest) via

.. code-block:: bash

    pip install -r requirements-dev.txt


Now the tests can be run by going to the root directory of your decotools repository and executing

.. code-block:: bash

    make tests

=====================
Step 8: Documentation
=====================

If necessary for your contribution, add the appropriate documentation to the files in the ``docs/`` directory

========================================
Step 9: Committing and uploading changes
========================================

Now the changes you've made are ready to be committed and uploaded to GitHub. Let git know which files you would like to include in your commit via

.. code-block:: bash

    $ git add <modifies_files_here>


and then commit your changes with

.. code-block:: bash

    $ git commit -m '<meaningful messages about the changes made>'


Now you can push this commit from your local repository to your copy on GitHub

.. code-block:: bash

    $ git push origin <new_feature_branch_name>


==================================
Step 10: Submitting a pull request
==================================

Finally, you can go to your copy of decotools on GitHub and submit a pull request by clicking the "Compare & pull request" button!

.. image:: _static/pull-request-button.png
