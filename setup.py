from distutils.core import setup
setup(
  name = 'CosinorPy'
  packages = ['CosinorPy'],  
  version = '0.1',      
  license='MIT',        
  description = 'Python package for cosinor based rhytmomethry',   
  author = 'Miha Moskon',                   
  author_email = 'miha.moskon@fri.uni-lj.si',      
  url = 'https://github.com/mmoskon/CosinorPy',   
  download_url = 'https://github.com/mmoskon/CosinorPy/archive/v_0.1.tar.gz',   
  keywords = ['cosinor', 'rhytmomethry', 'regression', 'bioinformatics'],  
  install_requires=[            
          'pandas',
          'numpy',
          'matplotlib',
          'statsmodels',
          'scipy',
          'openpyxl',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Researchers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)