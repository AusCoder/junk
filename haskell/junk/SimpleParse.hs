module SimpleParse where

import Control.Applicative
import Data.Char (digitToInt)
import Data.Either (either)
import Data.Functor.Identity (Identity, runIdentity)
import qualified Text.Parsec as TP
import qualified Text.Parsec.Char as TPC

type Parser s a = TP.ParsecT s () Identity a

number :: Parser String Int
number = fmap (numberFromDigits . fmap digitToInt) (TP.many1 TPC.digit)

numberFromDigits :: [Int] -> Int
numberFromDigits = foldl (\acc n -> acc * 10 + n) 0

-- main :: IO ()
-- main = do
--   let inputStr = "123a"
--   let parsed = runIdentity $ TP.runParserT number () "" inputStr
--   either print print parsed
