class students:
    def __init__(self,name,major,code_score,group_score,exam_score):
        self.name=name
        self.major=major
        self.code_score=code_score
        self.group_score=group_score
        self.exam_score=exam_score

Jay=students('Jay','BMI',99,98,99)
print("Jay's information:")
print(Jay.__dict__)
