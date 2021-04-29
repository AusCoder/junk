module Boxes where

data BoundingBox a = BoundingBox
  { x_min :: a
  }
  deriving (Show)

sample :: BoundingBox Int
sample = BoundingBox {x_min = 4}
