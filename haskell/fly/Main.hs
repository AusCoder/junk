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
    CursesException,
    CursorMode (CursorInvisible),
    Event (EventCharacter),
    Update,
    Window,
    catchCurses,
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

instance Show Window where
  show _ = "Window (there is probably more to it than that!)"

data GameState = GameState
  { xPosition :: Integer,
    yPosition :: Integer,
    playingState :: PlayingState,
    ticks :: Integer,
    screenHeight :: Integer,
    screenWidth :: Integer,
    spriteHeight :: Integer,
    tickDuration :: Integer
  }
  deriving (Show)

data GameEvent = Tick | UserEvent Event deriving (Show)

data GameEnv = GameEnv {getGameStateRef :: IORef GameState, getWindow :: Window}

initialGameEnv :: Curses GameEnv
initialGameEnv = do
  window <- defaultWindow
  (height, width) <- screenSize
  gsRef <-
    liftIO . newIORef $
      GameState
        { xPosition = 3,
          yPosition = 3,
          playingState = Running,
          ticks = 0,
          screenHeight = height,
          screenWidth = width,
          spriteHeight = 2,
          tickDuration = 100
        }
  return
    GameEnv
      { getGameStateRef = gsRef,
        getWindow = window
      }

main :: IO ()
main = runCurses $ do
  setEcho False
  setCursorMode CursorInvisible
  gameEnv <- initialGameEnv
  runReaderT mainLoop gameEnv

mainLoop :: ReaderT GameEnv Curses ()
mainLoop = do
  -- Main loop logic:
  -- draw
  -- wait + gather events
  -- update game state

  window <- asks getWindow
  gameState <- ask >>= liftIO . readIORef . getGameStateRef
  lift . updateWindow window $ do
    clear
    moveCursor 0 0
    drawString $ "Ticks: " ++ show (ticks gameState)
    drawBird gameState
    moveCursor 0 0
  lift render

  events <- gatherEvents
  mapReaderT liftIO $ updateGameState events
  playingState' <- ask >>= fmap playingState . liftIO . readIORef . getGameStateRef

  case playingState' of
    Running -> mainLoop
    GameOver -> return ()
    Exited -> return ()

updateGameState :: [GameEvent] -> ReaderT GameEnv IO ()
updateGameState events = do
  r <- asks getGameStateRef
  liftIO . modifyIORef r . fmap clipToScreen $
    flip (foldr applyEvent) events

applyEvent :: GameEvent -> GameState -> GameState
applyEvent Tick g@GameState {xPosition = x, yPosition = y, ticks = t} =
  g {xPosition = x + 1, yPosition = y + 1, ticks = t + 1}
applyEvent (UserEvent ev) g@GameState {yPosition = y} =
  case ev of
    EventCharacter 'q' -> g {playingState = Exited}
    EventCharacter ' ' -> g {yPosition = y - 3}
    EventCharacter 'w' -> g {yPosition = y - 3}
    _ -> g

clipToScreen :: GameState -> GameState
clipToScreen
  g@GameState
    { xPosition = x,
      yPosition = y,
      screenWidth = w,
      screenHeight = h,
      spriteHeight = sh
    } =
    g
      { yPosition = min (max 0 y) (h - sh),
        xPosition = if x < w - 5 then x else 0
      }

gatherEvents :: ReaderT GameEnv Curses [GameEvent]
gatherEvents =
  let loop :: [GameEvent] -> Integer -> ReaderT GameEnv Curses [GameEvent]
      loop acc dur = do
        window <- asks getWindow
        startTime <- liftIO getCurrentTime
        event <- lift . getEvent window $ Just dur
        endTime <- liftIO getCurrentTime
        let diffMillis = timeDiffMillis endTime startTime
        let acc' = maybe acc ((: acc) . UserEvent) event
        if diffMillis >= dur
          then return acc'
          else loop acc' (dur - diffMillis)
   in ask >>= fmap tickDuration . liftIO . readIORef . getGameStateRef >>= loop [Tick]

flapPosition :: GameState -> FlapPosition
flapPosition GameState {ticks = t} =
  case t `mod` 4 `div` 2 of
    0 -> FlapUp
    1 -> FlapDown

drawGame :: GameState -> Update ()
drawGame gameState = undefined

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

debug :: CursesException -> ReaderT GameEnv Curses ()
debug exc =
  let waitForQ :: ReaderT GameEnv Curses ()
      waitForQ = do
        window <- asks getWindow
        ev <- lift $ getEvent window Nothing
        case ev of
          Just (EventCharacter 'q') -> return ()
          _ -> waitForQ
   in do
        window <- asks getWindow
        gameState <- ask >>= liftIO . readIORef . getGameStateRef
        lift . updateWindow window $ do
          moveCursor 20 20
          drawString $ show exc
          moveCursor 21 20
          drawString $ show gameState
        lift render
        waitForQ
