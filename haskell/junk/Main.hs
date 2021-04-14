module Main where

import Control.Applicative
import Data.Char (digitToInt)
import Data.Functor.Identity (Identity)
import qualified Text.Parsec as TP
import qualified Text.Parsec.Char as TPC

type Parser s a = TP.ParsecT s () Identity a

number :: Parser String Int
number = fmap (numberFromDigits . fmap digitToInt) (TP.many TPC.digit)

numberFromDigits :: [Int] -> Int
numberFromDigits = foldl (\acc n -> acc * 10 + n) 0

main :: IO ()
main = do
  let inputStr = "123"
  print $ numberFromDigits [4, 5, 6, 2]
