module Main where

import qualified Data.Map.Strict as M
import Data.Maybe (fromMaybe)
import Data.Sort (sort)
import Debug.Trace (trace)

data Interval = Interval Int Int deriving (Show)

data Box = Box
  { xMin :: Int,
    yMin :: Int,
    xMax :: Int,
    yMax :: Int
  }

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

unionLength :: [Interval] -> Int
unionLength = sum . fmap (\(Interval x y) -> y - x) . disjointIntervals

-- this can be improved:
--  Use the hashmap, don't create key for every point, just endpoints
--  Even better, keep the running state of the y-intervals, for each
--  new point either remove an interval or add an interval
unionArea :: [Box] -> Int
unionArea boxes =
  let points = sort $ boxes >>= \b -> [xMin b, xMax b]
      boxContainsX x (Box a _ b _) = a <= x && x < b
      intervalAtX x (Box _ a _ b) = Interval a b

      boxesByX :: M.Map Int [Box]
      boxesByX = foldr (\x m -> M.insert x (filter (boxContainsX x) boxes) m) M.empty [minimum points .. maximum points]
      -- unionLengthAt x = unionLength . fmap (intervalAtX x) . fromMaybe undefined . M.lookup x $ boxesByX
      unionLengthAt x = unionLength . fmap (intervalAtX x) . filter (boxContainsX x) $ boxes
      foldFn (a, b) acc = unionLengthAt a * (b - a) + acc
   in foldr foldFn 0 $ zip points (tail points)

main :: IO ()
main = do
  -- print $ unionLength [Interval 3 4, Interval 3 7, Interval 6 9]

  print $ unionArea [Box 0 0 2 2, Box 2 0 6 2]
