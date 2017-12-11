import pygame
import sys
import numpy as np
import neat
import random

class Breakout:
    def __init__(self, genome=None, config=None, gui=True):
        self.net = None
        self.netLayersMap = None
        self.genome = genome
        self.config = config
        self.historyLength = 3
        self.drawExtraInfo = True
        self.maxGameLen = 60*60*5
        self.gameLen = 0
        if genome is not None and config is not None:
            self.net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.netLayersMap = self.getNetLayersMap()
        self.gui = gui
        if self.gui:
            self.screen = pygame.display.set_mode((800, 600))
        self.blocks = []
        self.paddle = [[pygame.Rect(300, 500, 20, 10), 120],
                [pygame.Rect(320, 500, 20, 10),100],
                [pygame.Rect(340, 500, 20, 10),80],
                [pygame.Rect(360, 500, 20, 10),45],
        ]
        self.ball = pygame.Rect(300, 490, 5, 5)
        self.speeds = {
            120:(-10, -3),
            100:(-10, -8),
            80:(10, -8),
            45:(10, -3),
        }
        self.swap = {
            120:45,
            45:120,
            100:80,
            80:100,
        }
        if self.gui:
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 25)

    def newGame(self):
        self.score = 0
        self.gameOver = False
        self.createBlocks()
        self.ball.x = 300
        self.ball.y = 490
        self.direction = -1
        self.yDirection = -1
        self.angle = 80
        self.l = 0
        self.r = 0
        self.ballxHistory = [self.ball.x] * self.historyLength
        self.ballyHistory = [self.ball.y] * self.historyLength
        self.paddlexHistory = [self.paddle[2][0].x] * self.historyLength

    def createBlocks(self):
        self.blocks = []
        self.blocksMap = {}
        y = 50
        for __ in range(200 // 10):
            x = 50
            for _ in range(800 // 25 - 6):
                block = pygame.Rect(x, y, 25, 10)
                self.blocks.append(block)
                self.blocksMap[(x,y)] = 1
                x += 27
            y += 12
        self.blocksMapKeys = sorted(self.blocksMap.keys())

    def ballUpdate(self):
        for _ in range(2):
            speed = self.speeds[self.angle]
            xMovement = True
            if _:
                self.ball.x += speed[0] * self.direction
            else:
                self.ball.y += speed[1] * self.direction * self.yDirection
                xMovement = False
            if self.ball.x <= 0 or self.ball.x >= 800:
                self.angle = self.swap[self.angle]
                if self.ball.x <= 0:
                    self.ball.x = 1
                else:
                    self.ball.x = 799
            if self.ball.y <= 0:
                self.ball.y = 1
                self.yDirection *= -1
            
            for paddle in self.paddle:
                if paddle[0].colliderect(self.ball):
                    self.angle = paddle[1]
                    self.direction = -1
                    self.yDirection = -1
                    break
            check = self.ball.collidelist(self.blocks)
            if check != -1:
                self.blocksMap[(self.blocks[check].x, self.blocks[check].y)]=0
                self.blocks.pop(check)

                if xMovement:
                    self.direction *= -1
                self.yDirection *= -1
                self.score += 1
            if self.ball.y > 600 or not self.blocks:
                self.gameOver = True

    def step(self):
        if self.net != None:
            self.paddleUpdateByNeuralNetwork()
        else:
            self.paddleUpdateByMouse()
        self.ballUpdate()
        self.ballxHistory.pop()
        self.ballxHistory.insert(0, self.ball.x)
        self.ballyHistory.pop()
        self.ballyHistory.insert(0, self.ball.y)
        self.paddlexHistory.pop()
        self.paddlexHistory.insert(0, self.paddle[2][0].x)
        self.gameLen += 1
        if self.gameLen == self.maxGameLen:
            self.gameOver = True
        #print(self.ball.x, self.paddle[0][0].x)

    def paddleOffset(self, offset):
        for p in self.paddle:
            p[0].x += offset
        if self.paddle[0][0].x < 0:
            self.paddle[0][0].x = 0
            self.paddle[1][0].x = 20
            self.paddle[2][0].x = 40
            self.paddle[3][0].x = 60
        elif self.paddle[3][0].x+20 > 800:
            self.paddle[0][0].x = 720
            self.paddle[1][0].x = 740
            self.paddle[2][0].x = 760
            self.paddle[3][0].x = 780

    def gameState(self):
        varInfo = [self.ball.x/800, self.ball.y/800, self.direction, self.yDirection, self.angle/360]
        varInfo += [p[0].x/800 for p in self.paddle]
        varInfo += [x/800 for x in self.ballxHistory]
        varInfo += [y/800 for y in self.ballyHistory]
        varInfo += [x/800 for x in self.paddlexHistory]
        #varInfo += [self.blocksMap[k] for k in self.blocksMapKeys]
        return np.array(varInfo, dtype=np.float64)

    def storePressedButtons(self, l, r):
        self.l, self.r = l,r

    def paddleUpdateByNeuralNetwork(self):
        l, r = self.net.activate(self.gameState())
        l = round(max(min(l,1),0))
        r = round(max(min(r,1),0))
        self.storePressedButtons(l,r)
        offset = 20*(r-l)
        self.paddleOffset(offset)

    def paddleUpdateByKeyboard(self):
        keys = pygame.key.get_pressed()
        l, r = keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
        self.storePressedButtons(l,r)
        offset = 20*(r-l)
        self.paddleOffset(offset)

    def paddleUpdateByMouse(self):
        pos = pygame.mouse.get_pos()
        on = 0
        for p in self.paddle:
            p[0].x = pos[0] + 20 * on
            on += 1

    def drawText(self, text, loc):
        self.screen.blit(self.font.render(str(text), -1, (255,255,255)), loc)

    def drawGame(self):
        self.screen.fill((0, 0, 0))
        for block in self.blocks:
            pygame.draw.rect(self.screen, (255,255,255), block)
        for paddle in self.paddle:
            pygame.draw.rect(self.screen, (255,255,255), paddle[0])
        pygame.draw.rect(self.screen, (255,255,255), self.ball)
        self.drawText(self.score, (400,520))
        x,y = 700, 300
        if self.drawExtraInfo:
            for info in self.gameState()[0:9]:
                self.drawText(round(info,2), (x,y))
                y += 30
            if self.l:
                self.drawText("L", (400, 550))
            if self.r:
                self.drawText("R", (430, 550))

    def getNetLayersMap(self):
        layers = {k:0 for k in self.config.genome_config.input_keys}
        connections = set()
        for cg in self.genome.connections.values():
            if cg.enabled:
                connections.add(cg.key)
        n = 0
        done = False
        while not done:
            done = True
            for a, b in connections:
                if a in layers and layers[a] == n:
                    layers[b] = n+1
                    done = False
            n += 1
        return layers


    def locationOnScreenOfInput(self, a):
        # input keys are -1 ,-2 , ..., -529
        # output keys are 0, 1
        if a < 0:
            a = abs(a)
            if a < 10:
                return 730, 300 + 30*(a-1)
            a -= 10
            x,y = self.blocksMapKeys[a]
            return x+12, y+5
        else:
            x, y = 800, 0
            numLayers = len(set(self.netLayersMap.values()))
            dx = 800//numLayers
            layer = self.netLayersMap[a]
            sameLayerInputs = sorted([b for b in self.netLayersMap if self.netLayersMap[b]==layer])
            numInLayer = len(sameLayerInputs)
            dy = 600//numInLayer
            i = sameLayerInputs.index(a)
            return x + dx*(layer-1), y + dy * i

    def drawConnection(self, a,b):
        start_pos = self.locationOnScreenOfInput(a)
        end_pos = self.locationOnScreenOfInput(b)
        pygame.draw.line(self.screen, (255,0,0), start_pos, end_pos)

    def drawNet(self):
        if self.net is None:
            return
        genome = self.genome
        config = self.config
        for cg in self.genome.connections.values():
            if cg.enabled:
                a,b = cg.key
                self.drawConnection(a,b)

    def play(self):
        if self.gui:
            #pygame.mouse.set_visible(False)
            clock = pygame.time.Clock()
        self.newGame()
        tickrate = 30
        while True:
            if self.gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return self.score
            if not self.gameOver:
                self.step()
            elif not self.gui: 
                return self.score
            if self.gui:
                clock.tick(tickrate)
                keys = pygame.key.get_pressed()
                u, d = keys[pygame.K_UP], keys[pygame.K_DOWN]
                if u:
                    tickrate += 20
                if d:
                    tickrate -= 20
                    tickrate = max(20,tickrate)

                self.drawGame()
                #self.drawNet()
                pygame.display.update()

if __name__ == "__main__":
    Breakout().play()



