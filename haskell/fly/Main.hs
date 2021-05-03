module Main where

import Control.Concurrent (threadDelay)
import Control.Monad (unless)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Reader
import Data.IORef (IORef (..), modifyIORef, newIORef, readIORef, writeIORef)
import Data.Maybe (maybeToList)
import Data.Ratio (denominator, numerator)
import Data.Time.Clock (UTCTime, diffUTCTime, getCurrentTime, nominalDiffTimeToSeconds)
import UI.NCurses
  ( Curses,
    CursorMode (CursorInvisible),
    Event (EventCharacter),
    Update,
    Window,
    clear,
    cursorPosition,
    defaultWindow,
    drawString,
    getEvent,
    moveCursor,
    render,
    runCurses,
    screenSize,
    setCursorMode,
    setEcho,
    updateWindow,
  )

data PlayingState = Running | GameOver | Exited deriving (Show)

data FlapPosition = FlapUp | FlapDown deriving (Show)

data GameState = GameState
  { xPosition :: Integer,
    yPosition :: Integer,
    playingState :: PlayingState,
    ticks :: Integer,
    screenHeight :: Integer,
    screenWidth :: Integer,
    spriteHeight :: Integer
  }
  deriving (Show)

data GameEvent = Tick | UserEvent Event deriving (Show)

newtype GameEnv = GameEnv {gameStateRef :: IORef GameState}

initialGameState :: Integer -> Integer -> GameState
initialGameState h w =
  GameState
    { xPosition = 3,
      yPosition = 3,
      playingState = Running,
      ticks = 0,
      screenHeight = h,
      screenWidth = w,
      spriteHeight = 2
    }

main :: IO ()
main = runCurses $ do
  setEcho False
  setCursorMode CursorInvisible
  window <- defaultWindow
  (height, width) <- screenSize
  gameEnv <-
    liftIO . fmap GameEnv . newIORef $ initialGameState height width

  mainLoop window gameEnv

mainLoop :: Window -> GameEnv -> Curses ()
mainLoop w gameEnv = do
  -- Main loop logic:
  -- draw
  -- wait + gather events
  -- update game state

  gameState <- liftIO . readIORef . gameStateRef $ gameEnv
  updateWindow w $ do
    clear
    moveCursor 0 0
    drawString . show . ticks $ gameState
    drawBird gameState
    moveCursor 0 0
  render

  events <- gatherEventsOverMillis w 100
  liftIO $ runReaderT (updateGameState events) gameEnv
  gameState' <- liftIO . readIORef . gameStateRef $ gameEnv

  case playingState gameState' of
    Running -> mainLoop w gameEnv
    GameOver -> return ()
    Exited -> return ()

updateGameState :: [GameEvent] -> ReaderT GameEnv IO ()
updateGameState events = do
  r <- asks gameStateRef
  liftIO . modifyIORef r $ flip (foldr applyEvent) events

applyEvent :: GameEvent -> GameState -> GameState
applyEvent Tick g@GameState {xPosition = x, yPosition = y, ticks = t} =
  clipToScreen $ g {xPosition = x + 1, yPosition = y + 1, ticks = t + 1}
applyEvent (UserEvent ev) g@GameState {yPosition = y} =
  case ev of
    EventCharacter 'q' -> g {playingState = Exited}
    EventCharacter ' ' -> clipToScreen $ g {yPosition = y - 3}
    EventCharacter 'w' -> clipToScreen $ g {yPosition = y - 3}
    _ -> g

clipToScreen :: GameState -> GameState
clipToScreen g@GameState {yPosition = y, screenHeight = h, spriteHeight = sh} =
  g {yPosition = min (max (-1) y) (h - sh)}

gatherEventsOverMillis :: Window -> Integer -> Curses [GameEvent]
gatherEventsOverMillis w dur =
  let loop dur' acc = do
        startTime <- liftIO getCurrentTime
        event <- getEvent w $ Just dur'
        endTime <- liftIO getCurrentTime
        let diffMillis = timeDiffMillis endTime startTime
        let acc' = acc ++ maybeToList (UserEvent <$> event)
        if diffMillis >= dur'
          then return acc'
          else loop (dur' - diffMillis) acc'
   in loop dur [Tick]

flapPosition :: GameState -> FlapPosition
flapPosition GameState {ticks = t} =
  case t `mod` 4 `div` 2 of
    0 -> FlapUp
    1 -> FlapDown

drawBird :: GameState -> Update ()
drawBird g@GameState {xPosition = x, yPosition = y} =
  let fPos = flapPosition g
   in moveCursor y x >> case fPos of
        FlapUp -> do
          drawString "\\   /"
          moveCursor (y + 1) x
          drawString " \\o/ "
        FlapDown -> do
          drawString " /o\\ "
          moveCursor (y + 1) x
          drawString "/   \\"

timeDiffMillis :: UTCTime -> UTCTime -> Integer
timeDiffMillis a b =
  let r = toRational . (* 1000) . nominalDiffTimeToSeconds $ diffUTCTime a b
   in numerator r `div` denominator r

-- waitFor :: Window -> (Event -> Bool) -> Curses ()
-- waitFor w p = loop
--   where
--     loop = do
--       ev <- getEvent w Nothing
--       case ev of
--         Nothing -> loop
--         Just ev' -> if p ev' then return () else loop
