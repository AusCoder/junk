module Main where

import Data.Sort (sort)

data Interval = Interval Int Int deriving (Show)

-- Takes a set of intervals, possibly overlapping
-- and returns a set of disjoint intervals spanning
-- the same set.
disjointIntervals :: [Interval] -> [Interval]
disjointIntervals intervals =
  let sortedPoints' = sort $ intervals >>= \(Interval x y) -> [(x, False), (y, True)]

      go inCount startPos acc [] = reverse acc
      go inCount startPos acc ((x, isEnd) : rest)
        | inCount == 0 = go 1 x acc rest
        | not isEnd = go (inCount + 1) startPos acc rest
        | otherwise =
          let newInCount = inCount - 1
           in if newInCount == 0
                then go newInCount undefined (Interval startPos x : acc) rest
                else go newInCount startPos acc rest
   in go 0 undefined [] sortedPoints'

main :: IO ()
main = do
  print $
    disjointIntervals
      [ Interval 1 5,
        Interval 6 10,
        Interval 2 4,
        Interval 12 14,
        Interval 7 9
      ]
