import os
import json
import shutil
import tempfile
import unittest
from steb.core import evaluate, get_model

class TestRetrieval(unittest.TestCase):
    def test_retrieval_dummy(self):
        model_name = "rrivera1849/LUAR-MUD"
        dataset = "dummy_retrieval"
        episode_size = 50
        n_episodes_per_class = 1
        
        # Create a temporary directory for outputs
        output_folder = tempfile.mkdtemp()
        
        try:
            model = get_model(model_name)
            
            evaluate(
                model,
                datasets=[dataset],
                episode_sizes=[episode_size],
                task_name="retrieval",
                n_episodes_per_class=n_episodes_per_class,
                force_reload=True,
                output_folder=output_folder,
                progress_bar=False
            )
            
            # Check results
            # Path: output_folder/dummy_retrieval/LUAR-MUD/50_1/retrieval/metrics.json
            model_str = os.path.basename(model_name)
            
            metrics_path = os.path.join(
                output_folder, 
                dataset, 
                model_str, 
                f"{episode_size}_{n_episodes_per_class}", 
                "retrieval", 
                "metrics.json"
            )
            
            self.assertTrue(os.path.exists(metrics_path), f"Metrics file not found at {metrics_path}")
            
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            print(f"Metrics: {metrics}")
            self.assertAlmostEqual(metrics["mrr"], 1.0, places=4, msg="MRR should be 1.0")
            
        finally:
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)

if __name__ == "__main__":
    unittest.main()
