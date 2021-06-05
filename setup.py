from setuptools import setup, find_packages
 
 
setup(name='pandasToBrat', 
      version='1.1.3',
      license='',
      author='Ali BELLAMINE',
      author_email='contact@alibellamine.me',
      description='Function for Brat folder administration from Python and Pandas object.',
      long_description=open('README.md').read(),
      packages = ["pandasToBrat", "pandasToBrat/extract_tools"]
    )
