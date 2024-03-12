import os

class Config_Path():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_filenames = []

    def get_model_filenames(self, folder_path):
        if not os.path.isdir(folder_path):
            raise ValueError("Folder path is not a valid directory.")

        filenames = []
        for filename in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, filename)):
                for ends in ['.bmodel']:
                    if filename.lower().endswith(ends):
                        filenames.append(filename)
                        break

        return filenames

    def update_all_model_names(self):
        self.model_filenames = self.get_model_filenames(self.model_path)


