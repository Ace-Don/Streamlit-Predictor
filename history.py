import copy
import datetime

class HistoryTracker:
    def __init__(self):
        self.history = []

    def save_version(self, dataset, action="Updated"):
        """Saves a snapshot of the dataset with a timestamp and action label."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        snapshot = {
            "timestamp": timestamp,
            "action": action,
            "data": copy.deepcopy(dataset)
        }
        self.history.append(snapshot)

    def get_versions(self):
        """Returns a list of saved versions."""
        return [f"{v['timestamp']} - {v['action']}" for v in self.history]

    def restore_version(self, index):
        """Restores a specific dataset version."""
        if 0 <= index < len(self.history):
            return self.history[index]['data']
        return None

    def has_history(self):
        """Checks if there are saved versions."""
        return len(self.history) > 0
