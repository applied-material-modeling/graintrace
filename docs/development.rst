Building & testing the docs
===========================

The documentation is built with Sphinx. The build needs none of the compiled
stack (NEML2, MOOSE/PUMA, NEPER, CUBIT); ``docs/conf.py`` mocks those imports,
so it runs from a plain checkout.

Install the docs toolchain
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"

Build the HTML
--------------

.. code-block:: bash

   cd docs
   make html
   # open docs/_build/html/index.html

For a strict build that treats warnings (broken cross-references, import
failures) as errors, the same check CI runs:

.. code-block:: bash

   sphinx-build -W --keep-going -b html docs docs/_build/html

Doc tests
---------

Runnable snippets marked with the ``doctest`` directive are executed by the
doctest builder, and external links are checked by the linkcheck builder:

.. code-block:: bash

   make -C docs doctest
   make -C docs linkcheck

A minimal doctest, verifying the package imports and exposes a version string:

.. doctest::

   >>> import graintrace
   >>> isinstance(graintrace.__version__, str)
   True

Continuous integration
----------------------

``.github/workflows/build_docs.yml`` runs the strict HTML build plus ``doctest``
and ``linkcheck`` on every push and pull request to ``main`` and
``documentation``. Pull requests get an ephemeral GitHub Pages preview; a push to
``main`` deploys the site to the ``gh-pages`` branch. The ``documentation`` branch
is built but never deployed.

Writing new pages
-----------------

- API pages under ``docs/api/`` are thin ``automodule`` stubs; add one per new
  public module and list it in ``docs/api/api.rst``.
- Tutorial pages under ``docs/tutorials/`` embed a runnable example with
  ``.. literalinclude:: ../../examples/<script>`` and are listed in
  ``docs/tutorials/tutorials.rst``.
- Keep the prose plain and factual; let the code examples carry the detail.

.. _definition-of-done:

Definition of done (adding or changing code)
--------------------------------------------

A new feature or code change is not complete until all five of these are done in
the same change:

1. **Tests pass.** Add or extend tests under ``tests/`` and keep the suite green:
   ``pytest``, ``black --check graintrace tests``, and
   ``pylint --rcfile=.pylintrc graintrace``. Tests that need the external stack
   self-skip via ``pytest.importorskip``/``skipif``.
2. **Docstrings updated.** Every new or changed public class, method, and
   function has a docstring (Google or NumPy style). Document new constructor
   keyword arguments; they surface directly in the API reference.
3. **Docs updated.** Update the relevant tutorial under ``docs/tutorials/``,
   :doc:`configuration` for new options, and add an ``automodule`` page under
   ``docs/api/`` for a new public module. The strict build and ``doctest`` must
   pass (see above).
4. **CLAUDE.md updated.** Update ``.claude/CLAUDE.md`` (the working reference for
   the codebase) so it stays the source of truth.
5. **MCP updated.** If the change is a workflow segment or adds/changes
   user-facing parameters, update the MCP tool in ``graintrace/mcp/tools/``, its
   recipe in ``graintrace/mcp/recipes/``, and the tool table in
   ``graintrace/mcp/README.md``. New external-tool dependencies must be reported
   by ``dependency_status``.
