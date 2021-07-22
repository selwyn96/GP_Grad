import papermill as pm


pm.execute_notebook('Main.ipynb',
                    'output.ipynb', 
                    kernel_name='papermill') 