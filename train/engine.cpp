#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Seg {
    int x;
    int y;
    bool operator==(const Seg& o) const { return x == o.x && y == o.y; }
};

// C++-owned snake state; Python training metadata (trajectory, state, action,
// value) is attached to the bound instance via py::dynamic_attr.
class Snake {
public:
    std::vector<Seg> body;  // head first
    int direction = 0;      // 0:+x 1:+y 2:-x 3:-y
    int survivalCount = 0;
    int foodCount = 0;
    int growCount = 0;
    int ateCount = 0;
    bool alive = true;
    bool ate = false;
    // Training metadata (trajectory, state, action, value) lives here so it is
    // shared by every Python wrapper of this C++ object and survives wrapper GC.
    // Lazily created as a dict on first access (never touched without the GIL).
    py::object meta;
};

// C++-owned game simulation state, mirroring train/game.py + train/food.py.
class Game {
public:
    Game(int w, int h)
        : gridWidth(w),
          gridHeight(h),
          maxFoodCount(w * h / 50),
          rng(std::random_device{}()) {}

    int gridWidth;
    int gridHeight;
    std::vector<std::unique_ptr<Snake>> snakes;
    std::vector<Seg> foods;
    std::vector<std::pair<int, Seg>> delayFoods;  // (ticks left, position)
    std::vector<int> delaySnakes;
    int maxFoodCount;
    int minFoodCount = 3;
    double deltaTime = 0;
    std::mt19937 rng;
};

struct DeathRec {
    int survival;
    int ate;
    int grow;
    int bodyLen;
};

bool isValidPos(const Game& g, int x, int y) {
    if (x < 0 || x >= g.gridWidth || y < 0 || y >= g.gridHeight) return false;
    for (const auto& s : g.snakes)
        for (const auto& seg : s->body)
            if (seg.x == x && seg.y == y) return false;
    for (const auto& f : g.foods)
        if (f.x == x && f.y == y) return false;
    return true;
}

Seg getValidPos(Game& g) {
    std::uniform_int_distribution<int> dx(0, g.gridWidth - 1);
    std::uniform_int_distribution<int> dy(0, g.gridHeight - 1);
    while (true) {
        Seg p{dx(g.rng), dy(g.rng)};
        if (isValidPos(g, p.x, p.y)) return p;
    }
}

std::unique_ptr<Snake> makeRandomSnake(Game& g) {
    Seg p = getValidPos(g);
    std::uniform_int_distribution<int> len(3, 32);
    auto s = std::make_unique<Snake>();
    s->body.assign(len(g.rng), p);
    return s;
}

void tickOne(Game& g, std::vector<DeathRec>& deaths) {
    // 1. move all snakes.
    for (auto& s : g.snakes) {
        if (!s->alive) continue;
        const int dx = s->direction == 0 ? 1 : s->direction == 2 ? -1 : 0;
        const int dy = s->direction == 1 ? 1 : s->direction == 3 ? -1 : 0;
        const Seg head{s->body[0].x + dx, s->body[0].y + dy};
        s->body.insert(s->body.begin(), head);
        s->body.pop_back();
    }

    // 2. collisions (wall / self / other snakes).
    for (auto& s : g.snakes) {
        if (!s->alive) continue;
        const Seg& head = s->body[0];
        bool dead = head.x < 0 || head.x >= g.gridWidth || head.y < 0 ||
                    head.y >= g.gridHeight;
        if (!dead) {
            for (size_t i = 1; i < s->body.size() && !dead; ++i) {
                if (s->body[i] == head) dead = true;
            }
        }
        if (!dead) {
            for (const auto& o : g.snakes) {
                if (o.get() == s.get() || !o->alive) continue;
                for (const auto& seg : o->body) {
                    if (seg == head) {
                        dead = true;
                        break;
                    }
                }
                if (dead) break;
            }
        }
        if (dead) {
            s->alive = false;
            deaths.push_back({s->survivalCount, s->ateCount, s->growCount,
                              static_cast<int>(s->body.size())});
        }
    }

    // 3. eat food / grow.
    for (auto& s : g.snakes) {
        if (!s->alive) continue;
        const Seg& head = s->body[0];
        for (auto it = g.foods.begin(); it != g.foods.end();) {
            if (it->x == head.x && it->y == head.y) {
                s->foodCount += 1;
                s->ateCount += 1;
                s->ate = true;
                it = g.foods.erase(it);
            } else {
                ++it;
            }
        }
        if (s->ate) {
            auto required = [&s]() {
                const size_t l = s->body.size();
                if (l < 10) return 1;
                if (l < 20) return 2;
                if (l < 30) return 3;
                if (l < 40) return 4;
                return 5;
            };
            while (s->foodCount >= required()) {
                s->foodCount -= required();
                s->body.push_back(s->body.back());
                s->growCount += 1;
            }
        }
    }

    // 4. respawn pending snakes.
    for (auto& d : g.delaySnakes) {
        d -= 1;
        if (d == 0) g.snakes.push_back(makeRandomSnake(g));
    }
    g.delaySnakes.erase(
        std::remove_if(g.delaySnakes.begin(), g.delaySnakes.end(),
                       [](int d) { return d == 0; }),
        g.delaySnakes.end());

    // 5. food manager (mirrors FoodManager.tick).
    const double foodRatio =
        double(g.foods.size() - g.minFoodCount) /
        double(g.maxFoodCount - g.minFoodCount);
    const double interval = 2 + 32 * foodRatio;
    g.deltaTime += 1;

    for (auto& df : g.delayFoods) {
        df.first -= 1;
        if (df.first == 0 && isValidPos(g, df.second.x, df.second.y)) {
            g.foods.push_back(df.second);
        }
    }
    g.delayFoods.erase(
        std::remove_if(g.delayFoods.begin(), g.delayFoods.end(),
                       [](const auto& df) { return df.first == 0; }),
        g.delayFoods.end());

    const bool spawn =
        (static_cast<int>(g.foods.size()) < g.maxFoodCount &&
         g.deltaTime >= interval) ||
        static_cast<int>(g.foods.size()) < g.minFoodCount;
    if (spawn) {
        g.deltaTime = std::max(0.0, g.deltaTime - interval);
        g.foods.push_back(getValidPos(g));
    }
}

void parallelFor(size_t n, size_t nthreads,
                 const std::function<void(size_t /*thread*/, size_t /*index*/)>& fn) {
    if (n == 0) return;
    nthreads = std::min(nthreads, n);
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (size_t t = 0; t < nthreads; ++t) {
        const size_t start = n * t / nthreads;
        const size_t end = n * (t + 1) / nthreads;
        threads.emplace_back([&fn, t, start, end]() {
            for (size_t i = start; i < end; ++i) fn(t, i);
        });
    }
    for (std::thread& th : threads) th.join();
}

unsigned hardwareThreads() {
    return std::max(1u, std::thread::hardware_concurrency());
}

std::vector<Game*> castGames(py::list games) {
    std::vector<Game*> out;
    out.reserve(games.size());
    for (py::handle h : games) out.push_back(h.cast<Game*>());
    return out;
}

py::list createGames(int count, int width, int height) {
    py::list out;
    for (int i = 0; i < count; ++i) {
        auto g = std::make_unique<Game>(width, height);
        for (int f = 0; f < 5; ++f) g->foods.push_back(getValidPos(*g));
        for (int s = 0; s < 5; ++s) g->snakes.push_back(makeRandomSnake(*g));
        out.append(py::cast(std::move(g)));
    }
    return out;
}

py::list tickGames(py::list games) {
    std::vector<Game*> gs = castGames(games);
    std::vector<std::vector<DeathRec>> deaths(gs.size());
    {
        py::gil_scoped_release release;
        parallelFor(gs.size(), hardwareThreads(), [&](size_t, size_t gi) {
            tickOne(*gs[gi], deaths[gi]);
        });
    }
    py::list out;
    for (auto& dvec : deaths) {
        for (const auto& d : dvec) {
            out.append(py::make_tuple(d.survival, d.ate, d.grow, d.bodyLen));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Observation building (moved from state_builder.cpp, now reads C++ state).
// ---------------------------------------------------------------------------

struct GameLayers {
    std::vector<float> snakesLayer;  // gridHeight * gridWidth
    std::vector<float> foodsLayer;   // gridHeight * gridWidth
};

void buildLayers(const Game& g, GameLayers& layers) {
    const size_t cells = static_cast<size_t>(g.gridWidth) * g.gridHeight;
    layers.snakesLayer.assign(cells, 0.0f);
    layers.foodsLayer.assign(cells, 0.0f);

    for (const auto& snake : g.snakes) {
        const size_t len = snake->body.size();
        for (size_t i = len; i-- > 0;) {
            const Seg& seg = snake->body[i];
            if (seg.x < 0 || seg.x >= g.gridWidth || seg.y < 0 ||
                seg.y >= g.gridHeight) {
                continue;  // dead snakes can hold out-of-grid heads
            }
            layers.snakesLayer[static_cast<size_t>(seg.y) * g.gridWidth + seg.x] =
                static_cast<float>(1.0 - 0.75 * static_cast<double>(i) /
                                            static_cast<double>(len));
        }
    }
    for (const Seg& f : g.foods) {
        layers.foodsLayer[static_cast<size_t>(f.y) * g.gridWidth + f.x] = 1.0f;
    }
}

void rot90Into(const float* src, float* dst, int k, int n, int stateSize) {
    k = (k % 4 + 4) % 4;
    const int last = n - 1;
    const int plane = n * n * stateSize;
    switch (k) {
        case 0:
            std::copy(src, src + plane, dst);
            break;
        case 1:
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int c = 0; c < stateSize; ++c)
                        dst[(i * n + j) * stateSize + c] =
                            src[(j * n + last - i) * stateSize + c];
            break;
        case 2:
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int c = 0; c < stateSize; ++c)
                        dst[(i * n + j) * stateSize + c] =
                            src[((last - i) * n + last - j) * stateSize + c];
            break;
        case 3:
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int c = 0; c < stateSize; ++c)
                        dst[(i * n + j) * stateSize + c] =
                            src[((last - j) * n + i) * stateSize + c];
            break;
    }
}

void buildState(const Game& g, const Snake& snake, const GameLayers& layers,
                int viewSize, int stateSize, float* scratch, float* out) {
    const int halfView = viewSize / 2;
    const int headX = snake.body[0].x;
    const int headY = snake.body[0].y;
    const int offsetX = halfView - headX;
    const int offsetY = halfView - headY;

    std::fill(scratch, scratch + static_cast<size_t>(viewSize) * viewSize * stateSize,
              0.0f);

    const size_t slen = snake.body.size();
    for (size_t i = slen; i-- > 0;) {
        const int x = snake.body[i].x + offsetX;
        const int y = snake.body[i].y + offsetY;
        if (x >= 0 && x < viewSize && y >= 0 && y < viewSize) {
            scratch[(static_cast<size_t>(y) * viewSize + x) * stateSize + 0] =
                static_cast<float>(1.0 - 0.75 * static_cast<double>(i) /
                                            static_cast<double>(slen));
        }
    }

    const int lvy = std::max(offsetY, 0);
    const int rvy = std::min(g.gridHeight + offsetY, viewSize);
    const int lvx = std::max(offsetX, 0);
    const int rvx = std::min(g.gridWidth + offsetX, viewSize);
    const int lwy = std::max(0, headY - halfView);
    const int rwy = std::min(g.gridHeight, headY + halfView + 1);
    const int lwx = std::max(0, headX - halfView);
    const int rwx = std::min(g.gridWidth, headX + halfView + 1);

    const int hView = rvy - lvy;
    const int wView = rvx - lvx;

    for (int j = 0; j < hView; ++j) {
        const float* srcRow =
            &layers.snakesLayer[static_cast<size_t>(lwy + j) * g.gridWidth + lwx];
        float* dstRow =
            &scratch[(static_cast<size_t>(lvy + j) * viewSize + lvx) * stateSize + 1];
        for (int i = 0; i < wView; ++i) dstRow[static_cast<size_t>(i) * stateSize] = srcRow[i];
    }

    if (hView != viewSize || wView != viewSize) {
        const size_t cells = static_cast<size_t>(viewSize) * viewSize;
        for (size_t idx = 0; idx < cells; ++idx) scratch[idx * stateSize + 2] = 1.0f;
        for (int j = 0; j < hView; ++j) {
            float* row =
                &scratch[(static_cast<size_t>(lvy + j) * viewSize + lvx) * stateSize + 2];
            for (int i = 0; i < wView; ++i) row[static_cast<size_t>(i) * stateSize] = 0.0f;
        }
    }

    for (int j = 0; j < hView; ++j) {
        const float* srcRow =
            &layers.foodsLayer[static_cast<size_t>(lwy + j) * g.gridWidth + lwx];
        float* dstRow =
            &scratch[(static_cast<size_t>(lvy + j) * viewSize + lvx) * stateSize + 3];
        for (int i = 0; i < wView; ++i) dstRow[static_cast<size_t>(i) * stateSize] = srcRow[i];
    }

    for (const Seg& f : g.foods) {
        int x = f.x + offsetX;
        int y = f.y + offsetY;
        if (x >= 0 && x < viewSize && y >= 0 && y < viewSize) continue;
        x -= halfView;
        y -= halfView;
        const double radius =
            std::sqrt(static_cast<double>(x) * x + static_cast<double>(y) * y);
        const int px = static_cast<int>(std::nearbyint(x / radius * halfView)) + halfView;
        const int py = static_cast<int>(std::nearbyint(y / radius * halfView)) + halfView;
        if (px >= 0 && px < viewSize && py >= 0 && py < viewSize) {
            float& cell = scratch[(static_cast<size_t>(py) * viewSize + px) * stateSize + 4];
            const double inc = (2.0 - std::min(2.0, (radius - halfView) / halfView)) / 10.0;
            cell = static_cast<float>(std::min(1.0, static_cast<double>(cell) + inc));
        }
    }

    rot90Into(scratch, out, snake.direction, viewSize, stateSize);
}

// L2 safety invariant: after the move, can the head still reach the tail?
// Classic snake result: if this holds after every move, the snake can never
// trap itself. The post-move tail occupies the cell of the old second-to-last
// segment (the old tail always vacates; a growth tick only duplicates the new
// tail in place), so the BFS goal is body[len-2] and the old tail cell is
// free. Other snakes are approximated as static bodies (minus their tails).
bool bfsCanReachTail(const Game& g, const Snake& snake, int headX, int headY,
                     std::vector<uint8_t>& blocked, std::vector<int>& queue) {
    const int W = g.gridWidth, H = g.gridHeight;
    std::fill(blocked.begin(), blocked.end(), 0);

    const size_t L = snake.body.size();
    for (size_t k = 0; k + 2 < L; ++k) {
        const Seg& sg = snake.body[k];
        if (sg.x >= 0 && sg.x < W && sg.y >= 0 && sg.y < H) {
            blocked[static_cast<size_t>(sg.y) * W + sg.x] = 1;
        }
    }
    for (const auto& o : g.snakes) {
        if (o.get() == &snake || !o->alive) continue;
        const size_t ol = o->body.size();
        for (size_t k = 0; k + 1 < ol; ++k) {
            const Seg& sg = o->body[k];
            if (sg.x >= 0 && sg.x < W && sg.y >= 0 && sg.y < H) {
                blocked[static_cast<size_t>(sg.y) * W + sg.x] = 1;
            }
        }
    }

    const Seg& goal = snake.body[L - 2];
    const int gx = goal.x, gy = goal.y;
    if (gx < 0 || gx >= W || gy < 0 || gy >= H) return false;
    blocked[static_cast<size_t>(gy) * W + gx] = 0;  // goal is the endpoint
    if (headX == gx && headY == gy) return true;

    queue.clear();
    const size_t startIdx = static_cast<size_t>(headY) * W + headX;
    blocked[startIdx] = 1;
    queue.push_back(static_cast<int>(startIdx));
    const int dxs[4] = {1, -1, 0, 0};
    const int dys[4] = {0, 0, 1, -1};
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int cur = queue[qi];
        const int cx = cur % W, cy = cur / W;
        for (int d = 0; d < 4; ++d) {
            const int nx = cx + dxs[d], ny = cy + dys[d];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            const size_t idx = static_cast<size_t>(ny) * W + nx;
            if (blocked[idx]) continue;
            if (nx == gx && ny == gy) return true;
            blocked[idx] = 1;
            queue.push_back(static_cast<int>(idx));
        }
    }
    return false;
}

void buildMask(const Game& g, const Snake& snake, uint8_t* out,
               std::vector<uint8_t>& blocked, std::vector<int>& queue) {
    const int headX = snake.body[0].x;
    const int headY = snake.body[0].y;
    for (int action = 0; action < 3; ++action) {
        const int dir = (snake.direction + action + 3) % 4;
        const int dx = (dir == 0) ? 1 : (dir == 2) ? -1 : 0;
        const int dy = (dir == 1) ? 1 : (dir == 3) ? -1 : 0;
        const int x = headX + dx;
        const int y = headY + dy;
        bool invalid = x < 0 || x >= g.gridWidth || y < 0 || y >= g.gridHeight;
        if (!invalid) {
            for (const auto& other : g.snakes) {
                for (size_t k = 0; k + 1 < other->body.size(); ++k) {
                    if (other->body[k].x == x && other->body[k].y == y) {
                        invalid = true;
                        break;
                    }
                }
                if (invalid) break;
            }
        }
        if (!invalid) {
            // L2: the move must leave the snake able to reach its own tail.
            invalid = !bfsCanReachTail(g, snake, x, y, blocked, queue);
        }
        out[action] = invalid ? 0 : 1;
    }
}

py::tuple buildStatesAndMasks(py::list games, int viewSize) {
    const int stateSize = 5;
    std::vector<Game*> gs = castGames(games);

    size_t totalSnakes = 0;
    for (const Game* g : gs) totalSnakes += g->snakes.size();

    py::array_t<float> states({static_cast<py::ssize_t>(totalSnakes),
                               static_cast<py::ssize_t>(viewSize),
                               static_cast<py::ssize_t>(viewSize),
                               static_cast<py::ssize_t>(stateSize)});
    py::array_t<uint8_t> masks(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(totalSnakes), 3});

    if (totalSnakes == 0) return py::make_tuple(states, masks);

    float* statesPtr = states.mutable_data();
    uint8_t* masksPtr = masks.mutable_data();
    const unsigned nthreads = hardwareThreads();

    std::vector<GameLayers> layers(gs.size());
    std::vector<size_t> offsets(gs.size() + 1, 0);
    for (size_t i = 0; i < gs.size(); ++i) {
        offsets[i + 1] = offsets[i] + gs[i]->snakes.size();
    }
    std::vector<std::vector<float>> scratches(
        nthreads, std::vector<float>(static_cast<size_t>(viewSize) * viewSize * stateSize));
    size_t maxCells = 0;
    for (const Game* g : gs) {
        maxCells = std::max(maxCells,
                            static_cast<size_t>(g->gridWidth) * g->gridHeight);
    }
    std::vector<std::vector<uint8_t>> blockedScratch(
        nthreads, std::vector<uint8_t>(maxCells));
    std::vector<std::vector<int>> queueScratch(nthreads);

    {
        py::gil_scoped_release release;
        parallelFor(gs.size(), nthreads, [&](size_t, size_t gi) {
            buildLayers(*gs[gi], layers[gi]);
        });
        parallelFor(totalSnakes, nthreads, [&](size_t threadIdx, size_t s) {
            const size_t gi =
                std::upper_bound(offsets.begin(), offsets.end(), s) - offsets.begin() - 1;
            const size_t si = s - offsets[gi];
            const Game& g = *gs[gi];
            const Snake& snake = *g.snakes[si];
            buildMask(g, snake, masksPtr + s * 3, blockedScratch[threadIdx],
                      queueScratch[threadIdx]);
            buildState(g, snake, layers[gi], viewSize, stateSize,
                       scratches[threadIdx].data(),
                       statesPtr + s * static_cast<size_t>(viewSize) * viewSize * stateSize);
        });
    }
    return py::make_tuple(states, masks);
}

py::list snakeBody(const Snake& s) {
    py::list out;
    for (const Seg& seg : s.body) out.append(py::make_tuple(seg.x, seg.y));
    return out;
}

void snakeSetBody(Snake& s, py::list body) {
    s.body.clear();
    s.body.reserve(body.size());
    for (py::handle h : body) {
        auto p = h.cast<std::pair<int, int>>();
        s.body.push_back({p.first, p.second});
    }
}

py::list gameSnakes(Game& g) {
    py::list out;
    for (auto& s : g.snakes)
        out.append(py::cast(s.get(), py::return_value_policy::reference));
    return out;
}

void gameSetSnakes(Game& g, py::list snakes) {
    g.snakes.clear();
    for (py::handle h : snakes) {
        auto* s = h.cast<Snake*>();
        g.snakes.push_back(std::make_unique<Snake>(*s));
    }
}

py::list gameFoods(const Game& g) {
    py::list out;
    for (const Seg& f : g.foods) out.append(py::make_tuple(f.x, f.y));
    return out;
}

void gameSetFoods(Game& g, py::list foods) {
    g.foods.clear();
    for (py::handle h : foods) {
        auto p = h.cast<std::pair<int, int>>();
        g.foods.push_back({p.first, p.second});
    }
}

}  // namespace

PYBIND11_MODULE(engine, m) {
    m.doc() = "Parallel C++ game engine and observation builder for ai-snake";

    m.def("create_games", &createGames, py::arg("count"), py::arg("width"),
          py::arg("height"));
    m.def("tick", &tickGames, py::arg("games"));
    m.def("build_states_and_masks", &buildStatesAndMasks, py::arg("games"),
          py::arg("view_size") = 23);
    m.def("hardware_threads", &hardwareThreads);

    py::class_<Snake>(m, "Snake", py::dynamic_attr())
        .def(py::init<>())
        .def_property("body", &snakeBody, &snakeSetBody)
        .def_property("direction", [](const Snake& s) { return s.direction; },
                      [](Snake& s, int v) { s.direction = (v % 4 + 4) % 4; })
        .def_property("alive", [](const Snake& s) { return s.alive; },
                      [](Snake& s, bool v) { s.alive = v; })
        .def_property("ate", [](const Snake& s) { return s.ate; },
                      [](Snake& s, bool v) { s.ate = v; })
        .def_property("survivalCount", [](const Snake& s) { return s.survivalCount; },
                      [](Snake& s, int v) { s.survivalCount = v; })
        .def_property("foodCount", [](const Snake& s) { return s.foodCount; },
                      [](Snake& s, int v) { s.foodCount = v; })
        .def_property("growCount", [](const Snake& s) { return s.growCount; },
                      [](Snake& s, int v) { s.growCount = v; })
        .def_property("ateCount", [](const Snake& s) { return s.ateCount; },
                      [](Snake& s, int v) { s.ateCount = v; })
        .def_property("meta",
                      [](Snake& s) -> py::object {
                          if (!s.meta) s.meta = py::dict();
                          return s.meta;
                      },
                      [](Snake& s, py::object v) { s.meta = v; })
        .def("setDirection", [](Snake& s, int action) {
            s.direction = (s.direction + action + 4) % 4;
            s.survivalCount += 1;
            s.ate = false;
        });

    py::class_<Game>(m, "Game", py::dynamic_attr())
        .def(py::init<int, int>())
        .def_property_readonly("gridWidth", [](const Game& g) { return g.gridWidth; })
        .def_property_readonly("gridHeight", [](const Game& g) { return g.gridHeight; })
        .def_property("snakes", &gameSnakes, &gameSetSnakes)
        .def_property("foods", &gameFoods, &gameSetFoods)
        .def_property_readonly("delaySnakes", [](const Game& g) {
            py::list out;
            for (int d : g.delaySnakes) out.append(d);
            return out;
        })
        .def_property_readonly("delayFoods", [](const Game& g) {
            py::list out;
            for (const auto& df : g.delayFoods)
                out.append(py::make_tuple(df.first, df.second.x, df.second.y));
            return out;
        })
        .def("add_delay_food",
             [](Game& g, int x, int y, int delay) { g.delayFoods.push_back({delay, {x, y}}); },
             py::arg("x"), py::arg("y"), py::arg("delay") = 6)
        .def("add_delay_snake", [](Game& g, int delay) { g.delaySnakes.push_back(delay); },
             py::arg("delay") = 20)
        .def("remove_dead_snakes", [](Game& g) {
            g.snakes.erase(
                std::remove_if(g.snakes.begin(), g.snakes.end(),
                               [](const auto& s) { return !s->alive; }),
                g.snakes.end());
        });
}
