module Main where

import Boxes
import qualified Graphics.Image as GI
import qualified Graphics.Image.Interface as GII
import System.IO

main :: IO ()
main = do
  ls <- readFile "/proc/modules"
  let moduleNames = map (head . words) $ lines ls
  mapM_ putStrLn moduleNames
  print sample

splitOn :: Eq a => a -> [a] -> [[a]]
splitOn c s = case dropWhile (== c) s of
  [] -> []
  s' ->
    let (w, s'') = break (== c) s'
     in w : splitOn c s''

-- main :: IO ()
-- main = do
--   image1 <- GI.readImageRGB GI.VU "/home/seb/Pictures/cricket.jpg"
--   image2 <- GI.readImageRGB GI.VU "/home/seb/Pictures/cricket.jpg"
--   print $ GII.foldr ((+) . abs) 0 $ image1 - image2
