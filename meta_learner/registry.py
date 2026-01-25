import os, json

class SkillRegistry:
    def __init__(self, skill_dir):
        self.dir = os.path.abspath(skill_dir)
        self.skills = self._discover()

    def _discover(self):
        skills = {}
        for root, _, files in os.walk(self.dir):
            for f in files:
                if f == "SKILL.md":
                    path = os.path.join(root, f)
                    name = os.path.basename(os.path.dirname(path))
                    with open(path) as fh:
                        skills[name] = fh.read()
        return skills

    def list(self):
        return list(self.skills.keys())
