from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = "Ajay"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5.0, description="A decimal value range between 0 to 10")

new_student = {"name": "Rohit", "age": 28, "email": "abc@docker.com", "cgpa": 8}

student  = Student(**new_student)

student_dict = student.model_dump()
print(student_dict['age'])