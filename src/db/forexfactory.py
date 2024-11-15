from peewee import Model, CharField, FloatField, IntegerField, DateTimeField, BooleanField
from peewee import SqliteDatabase

db = SqliteDatabase('db.db')

class BaseModel(Model):
    class Meta:
        database = db

class Economic(BaseModel):
    actual = FloatField()
    event = CharField()
    currency = CharField()
    date = DateTimeField()

# Create tables
db.connect()
db.create_tables([Economic])