import unittest
import yaml

class ImportYAMLTest(unittest.TestCase):
    """
    Unit tests for verifying the structure and contents of a YAML configuration file.

    This test suite includes the following tests:
    - Loading the YAML file once for all tests.
    - Ensuring top-level keys exist in the YAML file.
    - Checking that `dictionary_paths` contains the correct keys.
    - Verifying that `comorbidity_types` is a list of strings.
    - Verifying that `em_types` is a list of strings.
    - Verifying that `year_types` is a list of strings.

    Tests:
    - `test_yaml_structure`: Ensures the presence of top-level keys in the YAML file.
    - `test_dictionary_paths_keys`: Checks that `dictionary_paths` contains the expected keys.
    - `test_comorbidity_types`: Ensures `comorbidity_types` is a list of strings.
    - `test_em_types`: Ensures `em_types` is a list of strings.
    - `test_year_types`: Ensures `year_types` is a list of strings.
    """
    @classmethod
    def setUpClass(cls):
        # Load the YAML file once for all tests.
        with open("/Users/jackregan/Downloads/kamaleswaran-labs/Sepy/em_make_pickle_config.yaml", "r", encoding="utf-8") as file:
            cls.yaml_data = yaml.safe_load(file)

    def test_yaml_structure(self):
        """ Ensure top-level keys exist in the YAML file. """
        self.assertIn("yearly_instance", self.yaml_data)
        self.assertIn("dictionary_paths", self.yaml_data)
        self.assertIn("data_path", self.yaml_data)
        self.assertIn("groupings_path", self.yaml_data)
        self.assertIn("output_path", self.yaml_data)

    def test_dictionary_paths_keys(self):
        """ Check that dictionary_paths contains the correct keys. """
        expected_keys = {"comorbidity_types", "em_types", "year_types"}
        actual_keys = set(self.yaml_data["dictionary_paths"].keys())
        expected_keys = {"comorbidity_types", "em_types", "year_types"}
        self.assertEqual(expected_keys, actual_keys)

    def test_comorbidity_types(self):
        """ Ensure comorbidity_types is a list of strings. """
        comorbidity_types = self.yaml_data["dictionary_paths"]["comorbidity_types"]
        self.assertIsInstance(comorbidity_types, list)
        self.assertTrue(all(isinstance(item, str) for item in comorbidity_types))

    def test_em_types(self):
        """ Ensure em_types is a list of strings. """
        em_types = self.yaml_data["dictionary_paths"]["em_types"]
        self.assertIsInstance(em_types, list)
        self.assertTrue(all(isinstance(item, str) for item in em_types))

    def test_year_types(self):
        """ Ensure year_types is a list of strings. """
        year_types = self.yaml_data["dictionary_paths"]["year_types"]
        self.assertIsInstance(year_types, list)
        self.assertTrue(all(isinstance(item, str) for item in year_types))
 
    def test_import_encounters_keys(self):
        """ Test case to check if the keys within each import are present """
        import_encounters = self.yaml_data["yearly_instance"]["import_encounters"]
        self.assertIn("drop_cols", import_encounters)
        self.assertIn("index_col", import_encounters)
        self.assertIn("date_cols", import_encounters)

    def test_import_demographics_keys(self):
        """ Test case to check if the keys within each import are present """
        import_demographics = self.yaml_data["yearly_instance"]["import_demographics"]
        assert "drop_cols" in import_demographics
        assert "index_col" in import_demographics
        assert "date_cols" in import_demographics

    def test_import_infusion_meds_keys(self):
        """ Test case to check if the keys within each import are present """ 
        import_infusion_meds = self.yaml_data["yearly_instance"]["import_infusion_meds"]
        assert "drop_cols" in import_infusion_meds
        assert "numeric_cols" in import_infusion_meds
        assert "anti_infective_group_name" in import_infusion_meds
        assert "vasopressor_group_name" in import_infusion_meds
        assert "index_col" in import_infusion_meds
        assert "date_cols" in import_infusion_meds

    def test_import_labs_keys(self):
        """ Test case to check if the keys within each import are present """
        import_labs = self.yaml_data["yearly_instance"]["import_labs"]
        assert "drop_cols" in import_labs
        assert "group_cols" in import_labs
        assert "date_cols" in import_labs
        assert "index_col" in import_labs
        assert "numeric_cols" in import_labs

    def test_import_vitals_keys(self):
        """ Test case to check if the keys within each import are present """
        import_vitals = self.yaml_data["yearly_instance"]["import_vitals"]
        assert "drop_cols" in import_vitals
        assert "numeric_cols" in import_vitals
        assert "index_col" in import_vitals
        assert "date_cols" in import_vitals
        assert "merge_cols" in import_vitals
        
    def test_index_col_encounters(self):
        """ Test specific value checks, like checking if the columns are correct. """
        import_encounters = self.yaml_data["yearly_instance"]["import_encounters"]
        assert import_encounters["index_col"] == ["csn"]

    def test_date_cols_procedures(self):
        """ Test specific value checks, like checking if the columns are correct. """
        import_procedures = self.yaml_data["yearly_instance"]["import_procedures"]
        assert import_procedures["date_cols"] == ["surgery_date", "in_or_dttm", "procedure_start_dttm", "procedure_comp_dttm", "out_or_dttm"]

    def test_anti_infective_group_name(self):
        """ Test specific value checks, like checking if the columns are correct. """
        import_infusion_meds = self.yaml_data["yearly_instance"]["import_infusion_meds"]
        assert import_infusion_meds["anti_infective_group_name"] == "anti-infective"

if __name__ == "__main__":
    unittest.main()
