#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <cglm/cglm.h>
#include <time.h>
// #define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
// #define CIMGUI_USE_GLFW
// #define CIMGUI_USE_OPENGL3
// #include <cimgui.h>
// #include <cimgui_impl.h>

#define STR2(x) #x
#define STR(x) STR2(x)

#ifdef _WIN32
#define INCBIN_SECTION ".rdata, \"dr\""
#else
#define INCBIN_SECTION ".rodata"
#endif

#define INCBIN(name, file) \
	__asm__(".section " INCBIN_SECTION "\n" \
			".global incbin_" STR(name) "_start\n" \
			".balign 16\n" \
			"incbin_" STR(name) "_start:\n" \
			".incbin \"" file "\"\n" \
			\
			".global incbin_" STR(name) "_end\n" \
			".balign 1\n" \
			"incbin_" STR(name) "_end:\n" \
			".byte 0\n" \
	); \
	extern __attribute__((aligned(16))) const char incbin_ ## name ## _start[]; \
	extern                              const char incbin_ ## name ## _end[]

unsigned int mapVBO, mapVAO, quadVBO, quadVAO, mapEBO, fbo, rbo, depthBuffer, worldVBO, worldVAO, worldEBO;

const char *mapsVShaderSrc = 
		"#version 330 core\n"
		"layout (location = 0) in vec2 aPos;\n"
		"layout (location = 1) in vec2 aTexCoords;\n"
		"uniform vec2 position;\n"
		"uniform vec2 offset;\n"
		"uniform vec2 scale;\n"
		"uniform vec2 screen;\n"
		"out vec2 TexCoords;\n"
		"void main()\n"
		"{\n"
		" vec2 p = scale * (vec2(aPos.x, aPos.y) + 2. * offset) - 2. * position / screen;\n"
		" gl_Position = vec4(p.x, p.y, 0.0, 1.0);\n"
		" TexCoords = aTexCoords;\n"
		"}";
const char *mapsFShaderSrc = 
		"#version 330 core\n"
		"in vec2 TexCoords;\n"
		"uniform sampler2D texture0;\n"
		"uniform vec2 scale;\n"
		"uniform vec2 screen;\n"
		"in vec4 gl_FragCoord;\n"
		"out vec4 FragColor;\n"
		"void main()\n"
		"{\n"
		" FragColor=texture(texture0, TexCoords);"
		" if (scale.x > 0.02 && (TexCoords.x <= 1. / (scale.x * screen.x) || TexCoords.y <= 1. / (scale.y * screen.y))) FragColor+=vec4(.25);\n"
		"}";
const char *vertexShaderSrc = 
		"#version 330 core\n"
		"layout (location = 0) in vec2 aPos;\n"
		"layout (location = 1) in vec2 aTexCoords;\n"
		"out vec2 TexCoords;\n"
		"void main()\n"
		"{\n"
		" gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); \n"
		" TexCoords = aTexCoords;\n"
		"}";
const char *fragmentShaderSrc =
		"#version 330 core\n"
		"out vec4 FragColor;\n"
		"in vec2 TexCoords;\n"
		"in vec4 gl_FragCoord;\n"
		"uniform sampler2D screenTexture;\n"
		"uniform sampler2D depth;\n"
		"uniform mat4 invPersMatrix;\n"
		"uniform mat4 invViewMatrix;\n"
		"uniform mat4 persMatrix;\n"
		"uniform mat4 viewMatrix;\n"
		"uniform vec3 viewPos;\n"
		"uniform vec3 sun;\n"
		"uniform vec2 screen;\n"
		"uniform float fov;\n"
		"uniform int water;\n"
		"vec4 sky_colour = vec4(.4824, .5725, .9804, 1.);\n"
		"vec4 CalcEyeFromWindow(in vec3 windowSpace)\n"
		"{\n"
		" vec3 ndcPos;\n"
		" ndcPos.xy = (2.0 * windowSpace.xy) / (screen) - 1;\n"
		" ndcPos.z = 2.0 * windowSpace.z - 1.;\n"
		" \n"
		" vec4 clipPos;\n"
		" clipPos.w = persMatrix[3][2] / (ndcPos.z - (persMatrix[2][2] / persMatrix[2][3]));\n"
		" clipPos.xyz = ndcPos * clipPos.w;\n"
		" \n"
		" return invPersMatrix * clipPos;\n"
		"}\n"
		"\n"
		"void main()\n"
		"{\n"
		" float fogStart = 110.;\n"
		" float fogDensity = .0375;\n"
		" vec4 fog_colour = vec4(.6666, .8156, .9921, 1.);\n"
		" if (water == 1)\n"
		" {\n"
		"  fogStart = 0.;\n"
		"  fogDensity = .5;\n"
		"  fog_colour = vec4(.0196, .0196, .2, 1.);\n"
		" }\n"
		" vec2 uv = TexCoords;\n"
    " uv *=  1. - uv.yx;\n"
    " float vig = uv.x*uv.y * 15.0;\n"
    " vig = pow(vig, .125);\n"
		" float dist = texture(depth, TexCoords).x;\n"
		" \n"
		" vec4 eyeSpace = CalcEyeFromWindow(vec3(gl_FragCoord.x, gl_FragCoord.y, texture(depth, TexCoords).x));\n"
		" vec4 worldSpace = invViewMatrix * eyeSpace;\n"
		" if (dist == 1.)\n"
		" {\n"
		"  float pct = smoothstep(-.5,.5,worldSpace.y / 100. - 3.);\n"
		"  FragColor = mix(fog_colour, sky_colour, pct);\n"
		"  vec3 look = vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);\n"
		"  if (length(look - sun) < 1.) FragColor = vec4(1.);\n"
		" }\n"
		" else \n"
		" {\n"
		"  vec3 pos = viewPos;\n"
		"  float fragDist = length(pos - worldSpace.xyz);\n"
		"  float fog_factor = 1.0 - exp((fragDist-fogStart) * -fogDensity);\n"
		"  fog_factor = clamp(fog_factor, 0.0, 1.0);\n"
		"  FragColor = mix(texture(screenTexture, TexCoords), fog_colour, fog_factor);\n"
		" }\n"
		" FragColor = vec4(FragColor.xyz * vig, 1.);\n"
		"}";
const char *mapVShader =
		"#version 330 core\n"
		"layout(location=0) in vec3 aPos;\n"
		"layout(location=1) in vec2 aTexCoord;\n"
		"out vec2 TexCoord;\n"
		"uniform vec3 realPos;\n"
		"uniform vec3 scale;\n"
		"uniform vec3 offset;\n"
		"uniform float direction;\n"
		"void main()\n"
		"{\n"
		" vec3 p = aPos.xyz - 2. * realPos;\n"
    " float new_x = p.x*cos(direction) - p.y*sin(direction);\n"
    " float new_y = p.y*cos(direction) + p.x*sin(direction);\n"
		"	gl_Position = vec4(vec3(new_x, new_y, p.z) * scale + offset, 1.0);\n"
		"	TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
		"}";
const char *mapFShader =
		"#version 330 core\n"
		"in vec2 TexCoord;\n"
		"uniform sampler2D texture0;"
		"uniform vec3 realPos;\n"
		"out vec4 FragColor;\n"
		"void main()\n"
		"{\n"
		" vec2 pos = realPos.xy;"
		" if (length(TexCoord.xy - .5 - pos) > .4243) \n"
    " {\n"
    "  FragColor=vec4(0., 0., 0., 0.);\n"
    " }\n"
		" else if (length(TexCoord.xy - .5 - pos) > .4183) \n"
    " {\n"
    "  FragColor=vec4(1.);\n"
    " }\n"
		" else if (length(TexCoord.xy - .5 - pos) < .01) \n"
    " {\n"
    "  FragColor=vec4(1., 0., 0., 1.);\n"
    " }\n"
		" else\n"
		" {\n"
		"  FragColor=texture(texture0, TexCoord);\n"
		" }\n"
		"}";
const char *lightCasterVShader =
		"#version 330 core\n"
		"layout(location=0)in vec3 aPos;\n"
		"layout(location=1)in vec3 aNormal;\n"
		"layout(location=2)in vec2 aTexCoords;\n"
		"layout(location=3)in float aTextureID;\n"
		"out vec3 FragPos;\n"
		"out vec3 Normal;\n"
		"out vec2 TexCoords;\n"
		"out float TextureID;\n"
		"uniform mat4 model;\n"
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"
		"void main()\n"
		"{\n"
		" FragPos=vec3(model*vec4(aPos,1.));\n"
		" Normal=mat3(transpose(inverse(model)))*aNormal;\n"
		" TexCoords=aTexCoords;\n"
		" TextureID=aTextureID;\n"
		" gl_Position=projection*view*vec4(FragPos,1.);\n"
		"}";
const char *lightCasterFShader =
		"#version 330 core\n"
		"out vec4 FragColor;\n"
		"struct Material{\n"
		" float shininess;\n"
		"};\n"
		"struct Light{\n"
		" vec3 direction;\n"
		" vec3 ambient;\n"
		" vec3 diffuse;\n"
		" vec3 specular;\n"
		"};\n"
		"in vec3 FragPos;\n"
		"in vec3 Normal;\n"
		"in vec2 TexCoords;\n"
		"in float TextureID;\n"
		"in vec4 gl_FragCoord;\n"
		"uniform vec3 viewPos;\n"
		"uniform Material material;\n"
		"uniform sampler2DArray textures;\n"
		"uniform Light light;\n"
		"void main()\n"
		"{\n"
		" vec4 tex = texture(textures,vec3(TexCoords, TextureID));"
		" if (tex.a < .5) discard;\n"
		" vec3 ambient=light.ambient*tex.rgb;\n"
		" vec3 norm=normalize(Normal);\n"
		" vec3 lightDir=normalize(-light.direction);\n"
		" float diff=max(dot(norm,lightDir),0.);\n"
		" vec3 diffuse=light.diffuse*diff*tex.rgb;\n"
		" vec3 viewDir=normalize(viewPos-FragPos);\n"
		" vec3 reflectDir=reflect(-lightDir,norm);\n"
		" float spec=pow(max(dot(viewDir,reflectDir),0.),material.shininess);\n"
		" vec3 result=ambient+diffuse;\n"
		" FragColor = vec4(result, tex.a);\n"
		"}";

unsigned int shaderProgram[4];
unsigned int textures[1024];
unsigned char *texturesData[1024];
unsigned int texture, fboTexture, textureColorbuffer;
int textureCount = 0;
int shaderCount = 0;
unsigned int uniformView;
unsigned int uniformProjection;
unsigned int uniformModel;
unsigned int uniformRealPos;
unsigned int uniformOffset;
unsigned int uniformMapScale;
unsigned int uniformCameraDirection;
unsigned int uniformPitch;
unsigned int uniformYaw;
unsigned int uniformScreen;
unsigned int uniformView2;
unsigned int uniformInvView;
unsigned int uniformPers;
unsigned int uniformInvPers;
unsigned int uniformViewPos;
unsigned int uniformWater;
unsigned int uniformWorldMapScale;
unsigned int uniformWorldMapOffset;
unsigned int uniformWorldMapScreen;
unsigned int uniformWorldMapPos;

double lastTime;
int nbFrames = 0;
int fps = 0;
char title[100];

#define SCREEN_WIDTH_INIT 900
#define SCREEN_HEIGHT_INIT 800

int screenWidth = SCREEN_WIDTH_INIT;
int screenHeight = SCREEN_HEIGHT_INIT;

void loadShader(unsigned int vertexShader, unsigned int fragmentShader)
{
	shaderProgram[shaderCount] = glCreateProgram();

	glAttachShader(shaderProgram[shaderCount], vertexShader);
	glAttachShader(shaderProgram[shaderCount], fragmentShader);
	glLinkProgram(shaderProgram[shaderCount]);

	glUseProgram(shaderProgram[shaderCount]);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	++shaderCount;
}

static inline uint64_t
hashFunction(int8_t const *cstr, int len)
{
	uint64_t result = 14695981039346656037u;

	for (int i = 0; i < len; i++)
	{
		int64_t value = cstr[i];

		result ^= value;
		result *= 1099511628211u;
	}

	return result;
}

#define EXP 20
// Initialize all slots to an "empty" value (null)
#define HT_INIT \
	{		          \
		{0}, 0      \
	}

typedef struct ht
{
	uint64_t ht[1 << EXP];
	int htIdx[1 << EXP];
	int32_t len;
} ht;

int32_t ht_lookup(uint64_t hash, int exp, int32_t idx)
{
	uint32_t mask = ((uint32_t)1 << exp) - 1;
	uint32_t step = (hash >> (64 - exp)) | 1;
	return (idx + step) & mask;
}

CGLM_INLINE
int32_t getHash(int posX, int posZ)
{
	int64_t pos = (posX & 0xFFFFFFFFll) | ((uint64_t)(posZ) << 32);
	return hashFunction((int8_t *)&pos, sizeof(int64_t));
}

void addChunk(ht *t, const int posX, const int posZ, int index)
{
	int32_t h = getHash(posX, posZ);
	int32_t i = h;
	while (true)
	{
		i = ht_lookup(h, EXP, i);
		if (t->ht[i] == 0ull)
		{
			if ((uint32_t)t->len + 1 == (uint32_t)1 << EXP)
			{
				break;
			}
			t->len++;
			t->ht[i] = h;
			t->htIdx[i] = index;
			break;
		}
		else if (t->ht[i] == h)
		{
			break;
		}
	}
}

int hasChunk(ht *t, const int posX, const int posZ)
{
	int32_t h = getHash(posX, posZ);
	int32_t i = h;
	while (true)
	{
		i = ht_lookup(h, EXP, i);
		if (!t->ht[i])
		{
			return false;
		}
		else if (t->ht[i] == h)
		{
			return true;
		}
	}
}

int getChunkIdx(ht *t, const int posX, const int posZ)
{
	int32_t h = getHash(posX, posZ);
	int32_t i = h;
	while (true)
	{
		i = ht_lookup(h, EXP, i);
		if (!t->ht[i])
		{
			return -1;
		}
		else if (t->ht[i] == h)
		{
			return t->htIdx[i];
		}
	}
}

ht overworld = HT_INIT;

typedef int id[2];
typedef id chunk[16][256][16];

typedef bool FACES[6];

typedef struct meshCube
{
	ivec2 chunk;
	vec3 position;
	id ID;
	FACES faces;
} meshCube;

typedef struct chunkNode
{
	int index;
	int north;
	int south;
	int east;
	int west;
	int posX;
	int posZ;
	chunk chunk;
	int meshCount;
	int meshLimit;
	meshCube *meshCubes;
	int waterIndex;
	int verticesBufferCount;
	int verticesBufferLimit;
	float *verticesBuffer;
	unsigned int VBO;
	unsigned int VAO;
	int indicesBufferCount;
	int indicesBufferLimit;
	int *indicesBuffer;
	unsigned int EBO;
	float *heightMap;
	int *height;
	unsigned int texture;
} chunkNode;

int chunkCount = 0;
int chunkLimit = 1024;
chunkNode *chunks = NULL;
int chunksTVCount = 0;
int chunksIFCount = 0;
int chunksToView[1024];
int chunksInFront[1024];

typedef enum renderType {solid, transparent, empty, opaque} renderType;

typedef struct texturesId {
	int top;
	int down;
	int east;
	int west;
	int north;
	int south;
} texturesId;

typedef struct color {
	float r;
	float g;
	float b;
} color;

typedef struct block {
	renderType type;
	int textures[6];
	char *name;
	color averageColor;
} block;

block blocks[1024][1024];

bool testAab(mat4 MPV, vec3 min, vec3 max)
{
	float nxX = MPV[0][3] + MPV[0][0], nxY = MPV[1][3] + MPV[1][0], nxZ = MPV[2][3] + MPV[2][0], nxW = MPV[3][3] + MPV[3][0];
	float pxX = MPV[0][3] - MPV[0][0], pxY = MPV[1][3] - MPV[1][0], pxZ = MPV[2][3] - MPV[2][0], pxW = MPV[3][3] - MPV[3][0];
	float nyX = MPV[0][3] + MPV[0][1], nyY = MPV[1][3] + MPV[1][1], nyZ = MPV[2][3] + MPV[2][1], nyW = MPV[3][3] + MPV[3][1];
	float pyX = MPV[0][3] - MPV[0][1], pyY = MPV[1][3] - MPV[1][1], pyZ = MPV[2][3] - MPV[2][1], pyW = MPV[3][3] - MPV[3][1];
	float nzX = MPV[0][3] + MPV[0][2], nzY = MPV[1][3] + MPV[1][2], nzZ = MPV[2][3] + MPV[2][2], nzW = MPV[3][3] + MPV[3][2];
	float pzX = MPV[0][3] - MPV[0][2], pzY = MPV[1][3] - MPV[1][2], pzZ = MPV[2][3] - MPV[2][2], pzW = MPV[3][3] - MPV[3][2];
	
	return nxX * (nxX < 0 ? min[0] : max[0]) + nxY * (nxY < 0 ? min[1] : max[1]) + nxZ * (nxZ < 0 ? min[2] : max[2]) >= -nxW &&
		pxX * (pxX < 0 ? min[0] : max[0]) + pxY * (pxY < 0 ? min[1] : max[1]) + pxZ * (pxZ < 0 ? min[2] : max[2]) >= -pxW &&
		nyX * (nyX < 0 ? min[0] : max[0]) + nyY * (nyY < 0 ? min[1] : max[1]) + nyZ * (nyZ < 0 ? min[2] : max[2]) >= -nyW &&
		pyX * (pyX < 0 ? min[0] : max[0]) + pyY * (pyY < 0 ? min[1] : max[1]) + pyZ * (pyZ < 0 ? min[2] : max[2]) >= -pyW &&
		nzX * (nzX < 0 ? min[0] : max[0]) + nzY * (nzY < 0 ? min[1] : max[1]) + nzZ * (nzZ < 0 ? min[2] : max[2]) >= -nzW &&
		pzX * (pzX < 0 ? min[0] : max[0]) + pzY * (pzY < 0 ? min[1] : max[1]) + pzZ * (pzZ < 0 ? min[2] : max[2]) >= -pzW;
}

mat4 view = GLM_MAT4_IDENTITY_INIT;
mat4 invView;
mat4 projection;
mat4 invProjection;
float near = .1f;
float far = 1000.f;


// 100000.f
vec3 cameraPos = {100000.f, 80.f, 8.f};
vec3 cameraFront = {0.f, 0.f, -1.f};
vec3 cameraUp = {0.f, 1.f, 0.f};
vec3 cameraRight = {1.f, 0.f, 0.f};
ivec2 lastChunk;
ivec2 cameraChunk;
vec3 cameraChunkPos;
float cameraDirection;
float lastCameraDirection = 0;

float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;

bool cursorDisabled = true;
bool firstKeyE = true;
bool walked = false;
bool isOnWater = false;


bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = (float)SCREEN_WIDTH_INIT / 2.0;
float lastY = (float)SCREEN_HEIGHT_INIT / 2.0;

float fov = 70.0f;

float minimapScale = 250. / SCREEN_WIDTH_INIT;
bool mapView = false;
float mapScale = 50. / SCREEN_WIDTH_INIT;
bool firstKeyJ = true;

void calculateFrustum(void)
{
	chunksIFCount = 0;

	mat4 viewPos;
	mat4 MPV;
	glm_mat4_copy(view, viewPos);
	vec3 center;
	glm_vec3_add(cameraPos, cameraFront, center);
	glm_lookat(cameraPos, center, cameraUp, viewPos);
	glm_mat4_mul(projection, viewPos, MPV);

	for (int i = 0; i < chunksTVCount; i++)
	{
		chunkNode* c = &chunks[chunksToView[i]];

		vec3 min = {c->posX * 16, 0, c->posZ * 16};
		vec3 max = {c->posX * 16 + 16, 256, c->posZ * 16 + 16};

		if (testAab(MPV, min, max))
			chunksInFront[chunksIFCount++] = chunksToView[i];
	}
}

vec2 wherePressed = {0, 0};
bool canDrag = true;
vec2 mousePosition = {0, 0};
vec2 mapTranslation = {0, 0};
vec2 target = {0, 0};

void mouseCallback(GLFWwindow *window, double xposIn, double yposIn)
{
	if (mapView)
	{
		double x;
		double y;
		glfwGetCursorPos(window, &x, &y);
		glm_vec2_copy((vec2){x, y}, mousePosition);

		if (canDrag && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_RELEASE)
		{
			vec2 draged;
			glm_vec2_sub(target, mousePosition, draged);
			glm_vec2_add(wherePressed, draged, mapTranslation);
		}

		glUniform2f(uniformWorldMapPos, mapTranslation[0], -mapTranslation[1]);
	}

	if (!cursorDisabled)
		return;

	float xpos = xposIn;
	float ypos = yposIn;

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.1f; // change this value to your liking
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	// make sure that when pitch is out of bounds, screen doesn't get flipped
	if (pitch > 89.9f)
		pitch = 89.9f;
	if (pitch < -89.9f)
		pitch = -89.9f;

	vec3 front;
	front[0] = cos(glm_rad(yaw)) * cos(glm_rad(pitch));
	front[1] = sin(glm_rad(pitch));
	front[2] = sin(glm_rad(yaw)) * cos(glm_rad(pitch));
	glm_normalize(front);
	glm_vec3_copy(front, cameraFront);

	glm_vec3_cross(cameraFront, (vec3){0.f, 1.f, 0.f}, cameraRight);
	glm_vec3_normalize(cameraRight);

	glm_vec3_cross(cameraRight, cameraFront, cameraUp);
	glm_vec3_normalize(cameraUp);

	if(fabsf(lastCameraDirection - cameraDirection) > .01)
	{
		lastCameraDirection = cameraDirection;
		calculateFrustum();
	}
	
	glUseProgram(shaderProgram[1]);
	cameraDirection = atan2(cameraFront[2], cameraFront[0]);
	glUniform1f(uniformCameraDirection, cameraDirection);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	if (mapView){
		mapScale = mapScale * (yoffset / 2 + 1);
		if (mapScale > 0.45)
		{
			mapScale = 0.45;
			return;
		}

		vec2 halfScreen = {screenWidth / 2., screenHeight / 2.};
		vec2 vect1;
		vec2 vect2;

		glm_vec2_sub(mousePosition, halfScreen, vect1);
		glm_vec2_add(vect1, mapTranslation, vect2);
		glm_vec2_scale(vect2, yoffset / 2, vect2);
		glm_vec2_add(vect2, mapTranslation, mapTranslation);
		glm_vec2_add(vect1, mapTranslation, vect1);
		glm_vec2_copy(vect1, mapTranslation);

		glfwSetCursorPos(window, halfScreen[0], halfScreen[1]);

		glUniform2f(uniformWorldMapScale, mapScale, mapScale * (float)screenWidth / screenHeight);
		glUniform2f(uniformWorldMapPos, mapTranslation[0], -mapTranslation[1]);
	}
}

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		if (firstKeyE)
		{
			firstKeyE = false;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_RELEASE)
	{
		if (!firstKeyE)
		{
			firstKeyE = true;
			glfwSetCursorPos(window, (float)screenWidth / 2, (float)screenHeight / 2);
			lastX = (float)screenWidth / 2;
			lastY = (float)screenHeight / 2;

			if (cursorDisabled)
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
				cursorDisabled = false;
			}
			else
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				cursorDisabled = true;
			}
		}
	}

	if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
	{
		if (firstKeyJ)
		{
			firstKeyJ = false;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_J) == GLFW_RELEASE)
	{
		if (!firstKeyJ)
		{
			firstKeyJ = true;
			glfwSetCursorPos(window, (float)screenWidth / 2, (float)screenHeight / 2);
			lastX = (float)screenWidth / 2;
			lastY = (float)screenHeight / 2;

			if (mapView)
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				cursorDisabled = true;
				mapView = false;
			}
			else
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
				cursorDisabled = false;
				mapView = true;
			}
		}
	}

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = (5 * deltaTime);
	vec3 scale;
	vec3 cross;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		glm_vec3_scale(cameraFront, cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
		walked = true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		glm_vec3_scale(cameraFront, -cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
		walked = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		glm_cross(cameraFront, cameraUp, cross);
		glm_normalize(cross);
		glm_vec3_scale(cross, -cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
		walked = true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		glm_cross(cameraFront, cameraUp, cross);
		glm_normalize(cross);
		glm_vec3_scale(cross, cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
		walked = true;
	}
}

void resizeCallback(GLFWwindow *window, int width, int height)
{
	if (width == 0 || height == 0) return;
	calculateFrustum();
	screenWidth = width;
	screenHeight = height;
	glViewport(0, 0, width, height);
	glm_perspective(glm_rad(fov), (float)width / (float)height, 0.1f, 1000.0f, projection);

	glUseProgram(shaderProgram[0]);
	glUniformMatrix4fv(uniformProjection, 1, false, (float *)projection);

	glUseProgram(shaderProgram[1]);
	minimapScale = 250. / width;
	glUniform3f(uniformMapScale, minimapScale, (float)width / (float)height * minimapScale, 0);
	glUniform3f(uniformOffset, (width - 300.) / width, (height - 300.) / height, 0);

	glUseProgram(shaderProgram[2]);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, depthBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 0);
	glUniform2f(uniformScreen, width, height);
	glm_mat4_inv(projection, invProjection);
	glUniformMatrix4fv(uniformPers, 1, false, (float *)projection);
	glUniformMatrix4fv(uniformInvPers, 1, false, (float *)invProjection);

	glUseProgram(shaderProgram[3]);
	float scale = 50. / screenWidth;
	glUniform2f(uniformWorldMapScale, scale, (float)width / height * scale);
	glUniform2f(uniformWorldMapScreen, width, height);
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		glm_vec2_copy(mousePosition, target);
		glm_vec2_copy(mapTranslation, wherePressed);
		canDrag = true;
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		canDrag = false;
	}
}

bool hasAir(chunkNode *c, int x, int y, int z, int sideId[2])
{
	if (y >= 256)
		return true;

	if (x >= 16)
	{
		if (c->east >= 0)
		{
			x = 0;
			c = &chunks[c->east];
		}
		else
		{
			return false;
		}
	}
	else if (x < 0)
	{
		if (c->west >= 0)
		{
			x = 15;
			c = &chunks[c->west];
		}
		else
		{
			return false;
		}
	}
	if (z >= 16)
	{
		if (c->north >= 0)
		{
			z = 0;
			c = &chunks[c->north];
		}
		else
		{
			return false;
		}
	}
	else if (z < 0)
	{
		if (c->south >= 0)
		{
			z = 15;
			c = &chunks[c->south];
		}
		else
		{
			return false;
		}
	}

	if (y < 0)
	{
		return false;
	}

	int id[2];
	id[0] = c->chunk[x][y][z][0];
	id[1] = c->chunk[x][y][z][1];

	return id[0] == 0 || blocks[id[0]][id[1]].type == empty || blocks[id[0]][id[1]].type == transparent && sideId[0] != id[0];
}

void addFloat(chunkNode *c, float number)
{

	c->verticesBuffer[c->verticesBufferCount++] = number;

	if (c->verticesBufferCount + 1 >= c->verticesBufferLimit - 1)
	{
		c->verticesBufferLimit *= 2;
		c->verticesBuffer = (float *)realloc(c->verticesBuffer, sizeof(float) * c->verticesBufferLimit);
	}
}

int addVertex(chunkNode *c, float posX, float posY, float posZ, float normalX, float NormalY, float NormalZ, float textureCX, float textureCY, float textureID)
{
	addFloat(c, posX);
	addFloat(c, posY);
	addFloat(c, posZ);
	addFloat(c, normalX);
	addFloat(c, NormalY);
	addFloat(c, NormalZ);
	addFloat(c, textureCX);
	addFloat(c, textureCY);
	addFloat(c, textureID);

	return c->verticesBufferCount / 9 - 1;
}

void addIndex(chunkNode *c, int index)
{
	c->indicesBuffer[c->indicesBufferCount++] = index;

	if (c->indicesBufferCount + 1 >= c->indicesBufferLimit - 1)
	{
		c->indicesBufferLimit *= 2;
		c->indicesBuffer = (int *)realloc(c->indicesBuffer, sizeof(int) * c->indicesBufferLimit);
	}
}

void addIndices(chunkNode *c, int i1, int i2, int i3, int j1, int j2, int j3)
{
	addIndex(c, i1);
	addIndex(c, i2);
	addIndex(c, i3);
	addIndex(c, j1);
	addIndex(c, j2);
	addIndex(c, j3);
}

void generateVertices(chunkNode *chunkNodes, int init, int count)
{
	
	for (int i = init; i < count; i++)
	{
		chunkNode *c = &chunkNodes[i];

		c->verticesBufferCount = 0;
		c->indicesBufferCount = 0;

		int *indicesBlocks = NULL;
		int *indicesWater = NULL;
		int blockCount = 0;
		int waterCount = 0;

		indicesBlocks = (int *)malloc(sizeof(int) * c->meshCount);
		indicesWater = (int *)malloc(sizeof(int) * c->meshCount);

		// printf("limite: %i\n", c->meshCount);

		for (int j = 0; j < c->meshCount; j++)
		{
			meshCube *cube = &c->meshCubes[j];

			// printf("i: %i\n", j);

			if (cube->ID[0] == 5)
			{
				indicesWater[waterCount++] = j;
			}
			else
			{
				indicesBlocks[blockCount++] = j;
			}
		}

		int waterIndex = blockCount;
		// printf("agua: %i\n", waterCount);

		for (int j = 0; j < waterCount; j++)
		{
			indicesBlocks[blockCount++] = indicesWater[j];
		}

		float offset = 0;

		for (int j = 0; j < c->meshCount; j++)
		{
			meshCube *cube = &c->meshCubes[indicesBlocks[j]];
			vec3 pos;
			glm_vec3_copy(cube->position, pos);
			block b = blocks[cube->ID[0]][cube->ID[1]];
			int top = b.textures[0];
			int side = b.textures[2];
			int down = b.textures[1];

			if (j == waterIndex)
			{
				c->waterIndex = c->indicesBufferCount;
				offset = 2.f / 14;
			} 

			if (cube->faces[0])
			{
				int i1 = addVertex(c, pos[0] - .5, pos[1] + .5 - offset, pos[2] - .5, 0, 1, 0, 0, 0, top);
				int i2 = addVertex(c, pos[0] - .5, pos[1] + .5 - offset, pos[2] + .5, 0, 1, 0, 0, 1, top);
				int j1 = addVertex(c, pos[0] + .5, pos[1] + .5 - offset, pos[2] + .5, 0, 1, 0, 1, 1, top);
				int j2 = addVertex(c, pos[0] + .5, pos[1] + .5 - offset, pos[2] - .5, 0, 1, 0, 1, 0, top);

				addIndices(c, i1, i2, j1, j1, j2, i1);
				if (j >= waterIndex) addIndices(c, i1, j2, j1, j1, i2, i1);
			}
			if (cube->faces[1] && pos[1] > 0)
			{
				int i1 = addVertex(c, pos[0] - .5, pos[1] - .5, pos[2] - .5, 0, -1, 0, 0, 0, down);
				int i2 = addVertex(c, pos[0] + .5, pos[1] - .5, pos[2] - .5, 0, -1, 0, 1, 0, down);
				int j1 = addVertex(c, pos[0] + .5, pos[1] - .5, pos[2] + .5, 0, -1, 0, 1, 1, down);
				int j2 = addVertex(c, pos[0] - .5, pos[1] - .5, pos[2] + .5, 0, -1, 0, 0, 1, down);

				addIndices(c, i1, i2, j1, j1, j2, i1);
				if (j >= waterIndex) addIndices(c, i1, j2, j1, j1, i2, i1);
			}
			if (cube->faces[2])
			{
				int i1 = addVertex(c, pos[0] - .5, pos[1] - .5, pos[2] + .5, 0, 0, 1, 0, 0, side);
				int i2 = addVertex(c, pos[0] + .5, pos[1] - .5, pos[2] + .5, 0, 0, 1, 1, 0, side);
				int j1 = addVertex(c, pos[0] + .5, pos[1] + .5, pos[2] + .5, 0, 0, 1, 1, 1, side);
				int j2 = addVertex(c, pos[0] - .5, pos[1] + .5, pos[2] + .5, 0, 0, 1, 0, 1, side);

				addIndices(c, i1, i2, j1, j1, j2, i1);
				if (j >= waterIndex) addIndices(c, i1, j2, j1, j1, i2, i1);
			}
			if (cube->faces[3])
			{
				int i1 = addVertex(c, pos[0] - .5, pos[1] - .5, pos[2] - .5, 0, 0, -1, 0, 0, side);
				int i2 = addVertex(c, pos[0] - .5, pos[1] + .5, pos[2] - .5, 0, 0, -1, 0, 1, side);
				int j1 = addVertex(c, pos[0] + .5, pos[1] + .5, pos[2] - .5, 0, 0, -1, 1, 1, side);
				int j2 = addVertex(c, pos[0] + .5, pos[1] - .5, pos[2] - .5, 0, 0, -1, 1, 0, side);

				addIndices(c, i1, i2, j1, j1, j2, i1);
				if (j >= waterIndex) addIndices(c, i1, j2, j1, j1, i2, i1);
			}
			if (cube->faces[4])
			{
				int i1 = addVertex(c, pos[0] + .5, pos[1] - .5, pos[2] - .5, 1, 0, 0, 0, 0, side);
				int i2 = addVertex(c, pos[0] + .5, pos[1] + .5, pos[2] - .5, 1, 0, 0, 0, 1, side);
				int j1 = addVertex(c, pos[0] + .5, pos[1] + .5, pos[2] + .5, 1, 0, 0, 1, 1, side);
				int j2 = addVertex(c, pos[0] + .5, pos[1] - .5, pos[2] + .5, 1, 0, 0, 1, 0, side);

				addIndices(c, i1, i2, j1, j1, j2, i1);
				if (j >= waterIndex) addIndices(c, i1, j2, j1, j1, i2, i1);
			}
			if (cube->faces[5])
			{
				int i1 = addVertex(c, pos[0] - .5, pos[1] - .5, pos[2] - .5, -1, 0, 0, 0, 0, side);
				int i2 = addVertex(c, pos[0] - .5, pos[1] - .5, pos[2] + .5, -1, 0, 0, 1, 0, side);
				int j1 = addVertex(c, pos[0] - .5, pos[1] + .5, pos[2] + .5, -1, 0, 0, 1, 1, side);
				int j2 = addVertex(c, pos[0] - .5, pos[1] + .5, pos[2] - .5, -1, 0, 0, 0, 1, side);

				addIndices(c, i1, i2, j1, j1, j2, i1);
				if (j >= waterIndex) addIndices(c, i1, j2, j1, j1, i2, i1);
			}
		}
	}
}

void generateMesh(chunkNode *chunkNodes, int init, int count)
{
	for (int i = init; i < count; i++)
	{
		chunkNode *c = &chunkNodes[i];
		int posX = c->posX;
		int posZ = c->posZ;

		for (int x = 0; x < 16; x++)
		{
			for (int z = 0; z < 16; z++)
			{
				for (int y = 0; y < 256; y++)
				{
					meshCube cube = {
							.ID = {c->chunk[x][y][z][0], c->chunk[x][y][z][1]},
							.faces = {false, false, false, false, false, false},
							.chunk = {posX, posZ}};

					if (cube.ID[0] == 0)
						continue;

					if (hasAir(c, x, y + 1, z, cube.ID))
						cube.faces[0] = true;
					if (hasAir(c, x, y - 1, z, cube.ID))
						cube.faces[1] = true;
					if (hasAir(c, x, y, z + 1, cube.ID))
						cube.faces[2] = true;
					if (hasAir(c, x, y, z - 1, cube.ID))
						cube.faces[3] = true;
					if (hasAir(c, x + 1, y, z, cube.ID))
						cube.faces[4] = true;
					if (hasAir(c, x - 1, y, z, cube.ID))
						cube.faces[5] = true;

					for (int i = 0; i < 6; i++)
					{
						if (cube.faces[i])
						{
							vec3 position = {x, y, z};
							glm_vec3_copy(position, cube.position);
							c->meshCubes[c->meshCount++] = cube;

							if (c->meshCount + 1 >= c->meshLimit - 1)
							{
								c->meshLimit *= 2;
								c->meshCubes = (meshCube *)realloc(c->meshCubes, sizeof(meshCube) * c->meshLimit);
							}
							break;
						}
					}
				}
			}
		}
	}
}

float fract(float x)
{
	return x - floorf(x);
}

float random(vec2 st) {
	return fract(sinf(glm_vec2_dot(st, (vec2){12.9898, 78.233})) * 43759.5453123);
}

float mix(float x, float y, float a)
{
	return x * (1 - a) + y * a;
}

float noise(vec2 st)
{
	vec2 i = {floorf(st[0]), floorf(st[1])};
	vec2 f;
	f[0] = fract(st[0]);
	f[1] = fract(st[1]);

	// Four corners in 2D of a tile
	float a = random(i);
	vec2 b2;
	glm_vec2_add(i, (vec2){1, 0}, b2);
	float b = random(b2);
	vec2 c2;
	glm_vec2_add(i, (vec2){0, 1}, c2);
	float c = random(c2);
	vec2 d2;
	glm_vec2_add(i, (vec2){1, 1}, d2);
	float d = random(d2);

	vec2 u;

	vec2 square;
	glm_vec2_mul(f, f, square);
	vec2 doubleV;
	glm_vec2_scale(f, 2, doubleV);
	vec2 minus3;
	vec2 reverseDoubleV;
	glm_vec2_negate_to(doubleV, reverseDoubleV);
	glm_vec2_adds(reverseDoubleV, 3, minus3);
	glm_vec2_mul(square, minus3, u);

	return mix(a, b, u[0]) + (c - a) * u[1] * (1.0 - u[0]) + (d - b) * u[0] * u[1];
}

float fbm(vec2 st) {
	// Initial values
	float value = 0;
	float amplitude = .5;
	float frequency = 0;
	
	for (int i = 0; i < 6; i++) {
		vec2 input;
		glm_vec2_add(st, (vec2){i + 2, i + 1}, input);
		value += amplitude * noise(input);
		glm_vec2_scale(st, 2, st);
		amplitude *= .5;
	}

	return value;
}

void generateChunk(chunkNode *c, int posX, int posZ)
{
	if (chunkCount + 1 >= chunkLimit - 1)
	{
		chunkLimit *= 2;
		chunks = (chunkNode *)realloc(chunks, sizeof(chunkNode) * chunkLimit);
	}

	for (int x = 0; x < 16; x++)
	{
		for (int z = 0; z < 16; z++)
		{
			float k = 32;
			float px = (x + 16 * posX) / k;
			float pz = (z + 16 * posZ) / k;
			float noised = fbm((vec2){px, pz});

			int height = floor(20 * noised + 50);
			if (height > 59)
			{
				c->height[x * 16 + z] = height;
			}
			else
			{
				c->height[x * 16 + z] = 59;
			}

			for (int y = 0; y < 256; y++)
			{
				if (y < height - 5)
				{
					c->chunk[x][y][z][0] = 1;
					c->chunk[x][y][z][1] = 0;
				}
				else if (y < height)
				{
					c->chunk[x][y][z][0] = y <= 60 ? 6 : 2;
					c->chunk[x][y][z][1] = 0;
				}
				else if (y == height)
				{
					c->chunk[x][y][z][0] = y <= 60 ? 6 : 2;
					c->chunk[x][y][z][1] = y <= 60 ? 0 : 1;
				}
				else if (y <= 59)
				{
					c->chunk[x][y][z][0] = 5;
					c->chunk[x][y][z][1] = 0;
				}
				else
				{
					break;
				}
			}
		}
	}
}

void generateChunksMap(chunkNode *chunkNodes, int init, int count)
{
	for (int i = init; i < count; i++)
	{
		chunkNode *c = &chunkNodes[i];
		for (int x = 0; x < 16; x++)
		{
			for (int z = 0; z < 16; z++)
			{
				chunkNode* cToCompare = c;
				int id[2];
				glm_ivec2_copy(c->chunk[x][c->height[x * 16 + z]][z], id);
				color blockColor = blocks[id[0]][id[1]].averageColor;
				int height = c->height[x * 16 + z];
				int northHeight;
				int southHeight;
				int eastHeight;
				int westHeight;

				if (x == 0)
				{
					southHeight = c->height[(x + 1) * 16 + z];
					if (c->west >= 0)
					{
						northHeight = chunks[c->west].height[15 * 16 + z];
					}
					else
					{
						northHeight = height;
					}
				}
				else if (x == 15)
				{
					if (c->east >= 0)
					{
						southHeight = chunks[c->east].height[z];
					}
					else
					{
						southHeight = height;
					}
					northHeight = c->height[(x - 1) * 16 + z];
				}
				else
				{
					southHeight = c->height[(x + 1) * 16 + z];
					northHeight = c->height[(x - 1) * 16 + z];
				}

				if (z == 0)
				{
					eastHeight = c->height[x * 16 + z + 1];
					if (c->south >= 0)
					{
						westHeight = chunks[c->south].height[x * 16 + 15];
					}
					else
					{
						westHeight = height;
					}
				}
				else if (z == 15)
				{
					if (c->north >= 0)
					{
						eastHeight = chunks[c->north].height[x * 16];
					}
					else
					{
						eastHeight = height;
					}
					westHeight = c->height[x * 16 + z - 1];
				}
				else
				{
					eastHeight = c->height[x * 16 + z + 1];
					westHeight = c->height[x * 16 + z - 1];
				}

				if (southHeight > c->height[x * 16 + z])
				{
					blockColor.r /= 2;
					blockColor.g /= 2;
					blockColor.b /= 2;
				}
				else if (westHeight > c->height[x * 16 + z])
				{
					blockColor.r /= 1.5;
					blockColor.g /= 1.5;
					blockColor.b /= 1.5;
				}

				if (northHeight > c->height[x * 16 + z])
				{
					blockColor.r *= 1.5;
					blockColor.g *= 1.5;
					blockColor.b *= 1.5;
				}
				else if (eastHeight > c->height[x * 16 + z])
				{
					blockColor.r *= 1.25;
					blockColor.g *= 1.25;
					blockColor.b *= 1.25;
				}

				c->heightMap[(x * 16 + z) * 3] = blockColor.r / 255;
				c->heightMap[(x * 16 + z) * 3 + 1] = blockColor.g / 255;
				c->heightMap[(x * 16 + z) * 3 + 2] = blockColor.b / 255;
			}
		}

		glGenTextures(1, &c->texture);
		glBindTexture(GL_TEXTURE_2D, c->texture); 
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 16, 16, 0, GL_RGB, GL_FLOAT, c->heightMap);
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

unsigned int megaMapTexture;

void generateMiniMap(ht *dimension, int posX, int posZ)
{
	int metaX = (int)floor((float)posX / 16);
	int metaY = (int)floor((float)posZ / 16);
	float *megaTexture;
	megaTexture = (float *)malloc(256 * 256 * 3 * sizeof(float));

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			bool chunkGenerated = hasChunk(dimension, posX + i - 8, posZ + j - 8);
			chunkNode c;
			if (chunkGenerated)
			{
				c = chunks[getChunkIdx(dimension, posX + i - 8, posZ + j - 8)]; 
			}
			for (int x = 0; x < 16; x++)
			{
				for (int z = 0; z < 16; z++)
				{
					if (chunkGenerated)
					{
						megaTexture[((i * 16 + x) * 256 + j * 16 + z) * 3] = c.heightMap[(x * 16 + z) * 3];
						megaTexture[((i * 16 + x) * 256 + j * 16 + z) * 3 + 1] = c.heightMap[(x * 16 + z) * 3 + 1];
						megaTexture[((i * 16 + x) * 256 + j * 16 + z) * 3 + 2] = c.heightMap[(x * 16 + z) * 3 + 2];
					}
					if (x == 0 || z == 0)
					{
						megaTexture[((i * 16 + x) * 256 + j * 16 + z) * 3] += .2;
						megaTexture[((i * 16 + x) * 256 + j * 16 + z) * 3 + 1] += .2;
						megaTexture[((i * 16 + x) * 256 + j * 16 + z) * 3 + 2] += .2;
					}
				}
			}
		}
	}

	glGenTextures(1, &megaMapTexture);
	glBindTexture(GL_TEXTURE_2D, megaMapTexture); 

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 256, 256, 0, GL_RGB, GL_FLOAT, megaTexture);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void generateChunkSides(ht *dimension, chunkNode *chunkNodes, int init, int count)
{
	for (int i = init; i < count; i++)
	{
		int sideIndex;
		chunkNode *chunkPtr = &chunkNodes[i];
		int posX = chunkPtr->posX;
		int posZ = chunkPtr->posZ;
		if ((sideIndex = getChunkIdx(dimension, posX + 1, posZ)) >= 0)
		{
			chunkPtr->east = sideIndex;
			chunks[sideIndex].west = i;
		}
		if ((sideIndex = getChunkIdx(dimension, posX - 1, posZ)) >= 0)
		{
			chunkPtr->west = sideIndex;
			chunks[sideIndex].east = i;
		}
		if ((sideIndex = getChunkIdx(dimension, posX, posZ + 1)) >= 0)
		{
			chunkPtr->north = sideIndex;
			chunks[sideIndex].south = i;
		}
		if ((sideIndex = getChunkIdx(dimension, posX, posZ - 1)) >= 0)
		{
			chunkPtr->south = sideIndex;
			chunks[sideIndex].north = i;
		}
	}
}

void generateChunkNode(ht *dimension, const int posX, const int posZ, chunkNode **chunkNodes, int *count, int *chunkLimit)
{
	if (hasChunk(dimension, posX, posZ))
		return;

	if (*count + 2 >= *chunkLimit - 1)
	{
		*chunkLimit *= 2;
		*chunkNodes = (chunkNode *)realloc(*chunkNodes, sizeof(chunkNode) * (*chunkLimit));
	}
	chunkNode *chunkPtr = &(*chunkNodes)[(*count)++];

	chunkPtr->meshCount = 0;
	chunkPtr->verticesBufferCount = 0;
	chunkPtr->meshLimit = 1024;
	chunkPtr->verticesBufferLimit = 9 * 1024;
	chunkPtr->indicesBufferLimit = 6 * 1024;

	chunkPtr->meshCubes = (meshCube *)malloc(chunkPtr->meshLimit * sizeof(meshCube));
	chunkPtr->verticesBuffer = (float *)malloc(chunkPtr->verticesBufferLimit * sizeof(float));
	chunkPtr->indicesBuffer = (int *)malloc(chunkPtr->indicesBufferLimit * sizeof(int));

	chunkPtr->index = *count - 1;
	chunkPtr->north = -1;
	chunkPtr->south = -1;
	chunkPtr->east = -1;
	chunkPtr->west = -1;
	chunkPtr->posX = posX;
	chunkPtr->posZ = posZ;

	chunkPtr->heightMap = (float *)malloc(16 * 16 * 4 * sizeof(float));
	chunkPtr->height = (int *)malloc(16 * 16 * sizeof(int));

	chunkPtr->waterIndex = -1;

	memset(chunkPtr->chunk, 0, sizeof(chunkPtr->chunk));

	generateChunk(chunkPtr, posX, posZ);
	addChunk(dimension, posX, posZ, chunkPtr->index);
}

void addVBO(chunkNode *c)
{
	glGenVertexArrays(1, &c->VAO);
	glGenBuffers(1, &c->VBO);
	glGenBuffers(1, &c->EBO);

	glBindBuffer(GL_ARRAY_BUFFER, c->VBO);
	glBufferData(GL_ARRAY_BUFFER, c->verticesBufferCount * sizeof(float), c->verticesBuffer, GL_STATIC_DRAW);

	glBindVertexArray(c->VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, c->EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, c->indicesBufferCount * sizeof(int), c->indicesBuffer, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(8 * sizeof(float)));
	glEnableVertexAttribArray(3);
}

int viewDistance = 11;

void generateManyChunks(ht *dimension, int posX, int posZ, chunkNode **chunkNodes, int *count, int *chunkLimit)
{
	for (int i = posX - viewDistance; i <= posX + viewDistance; i++)
	{
		for (int j = posZ - viewDistance; j <= posZ + viewDistance; j++)
		{
			generateChunkNode(dimension, i, j, chunkNodes, count, chunkLimit);
		}
	}
}

void atualizeMovement(void)
{
	cameraChunk[0] = (int)floor(cameraPos[0] / 16);
	cameraChunk[1] = (int)floor(cameraPos[2] / 16);
	cameraChunkPos[0] = cameraPos[0] - cameraChunk[0] * 16;
	cameraChunkPos[1] = cameraPos[1];
	cameraChunkPos[2] = cameraPos[2] - cameraChunk[1] * 16;
	if (lastChunk[0] != cameraChunk[0] || lastChunk[1] != cameraChunk[1])
	{
		int lastChunkCount = chunkCount;

		int dx = cameraChunk[0] - lastChunk[0];
		int dy = cameraChunk[1] - lastChunk[1];

		for (int i = cameraChunk[0] - viewDistance; i < cameraChunk[0] + viewDistance; i++)
		{
			for (int j = cameraChunk[1] - viewDistance; j <= cameraChunk[1] + viewDistance; j++)
			{
				generateChunkNode(&overworld, i, j, &chunks, &chunkCount, &chunkLimit);
			}
		}

		if (dx != 0)
		{
			int signX = dx > 0 ? 1 : -1;

			int posX = cameraChunk[0] + signX * viewDistance - 2 * signX;
			for (int i = cameraChunk[1] - viewDistance; i <= cameraChunk[1] + viewDistance; i++)
			{
				int index = getChunkIdx(&overworld, posX, i);
				if (index >= 0)
				{
					chunks[index].meshCount = 0;
					chunks[index].verticesBufferCount = 0;
					generateMesh(chunks, index, index + 1);
					generateVertices(chunks, index, index + 1);
					addVBO(&chunks[index]);
				}
			}
		}

		if (dy != 0)
		{
			int signY = dy > 0 ? 1 : -1;
			int posZ = cameraChunk[1] + signY * viewDistance - 2 * signY;

			for (int i = cameraChunk[0] - viewDistance; i <= cameraChunk[0] + viewDistance; i++)
			{
				int index = getChunkIdx(&overworld, i, posZ);
				if (index >= 0)
				{
					chunks[index].meshCount = 0;
					chunks[index].verticesBufferCount = 0;
					generateMesh(chunks, index, index + 1);
					generateVertices(chunks, index, index + 1);
					addVBO(&chunks[index]);
				}
			}
		}

		generateChunkSides(&overworld, chunks, lastChunkCount, chunkCount);
		generateMesh(chunks, lastChunkCount, chunkCount);
		generateVertices(chunks, lastChunkCount, chunkCount);
		generateChunksMap(chunks, lastChunkCount, chunkCount);
		generateMiniMap(&overworld, cameraChunk[0], cameraChunk[1]);

		for (int i = lastChunkCount; i <= chunkCount; i++)
		{
			chunkNode *c = &chunks[i];
			addVBO(c);
		}

		chunksTVCount = 0;
		for (int i = cameraChunk[0] - viewDistance; i <= cameraChunk[0] + viewDistance; i++)
		{
			for (int j = cameraChunk[1] - viewDistance; j <= cameraChunk[1] + viewDistance; j++)
			{
				int idx = getChunkIdx(&overworld, i, j);
				if (idx >= 0 && idx <= chunkCount)
				{
					chunksToView[chunksTVCount++] = idx;
				} 
			}
		}

		lastChunk[0] = cameraChunk[0];
		lastChunk[1] = cameraChunk[1];
	}
}

// ImGuiIO *ioptr;

// void gui_init(GLFWwindow *window)
// {
// 	/* Initialize CIMGUI */
// 	// GL 3.2 + GLSL 130
// 	const char *glsl_version = "#version 130";

// 	igCreateContext(NULL);

// 	// set docking
// 	ioptr = igGetIO();
// 	ioptr->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
// // ioptr->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
// #ifdef IMGUI_HAS_DOCK
// 	ioptr->ConfigFlags |= ImGuiConfigFlags_DockingEnable;		// Enable Docking
// 	ioptr->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
// #endif

// 	ImGui_ImplGlfw_InitForOpenGL(window, true);
// 	ImGui_ImplOpenGL3_Init(glsl_version);

// 	igStyleColorsDark(NULL);
// }

// void gui_terminate(void)
// {
// 	ImGui_ImplOpenGL3_Shutdown();
// 	ImGui_ImplGlfw_Shutdown();
// 	igDestroyContext(NULL);
// }

// void gui_render(GLFWwindow *window)
// {
// 	igRender();
// 	glfwMakeContextCurrent(window);
// 	ImGui_ImplOpenGL3_RenderDrawData(igGetDrawData());
// }

// int position[3] = {0, 0, 0};
// double currentTime;
// int toHash[2] = {0, 0};

// void gui_update(GLFWwindow *window)
// {
// 	// start imgui frame
// 	ImGui_ImplOpenGL3_NewFrame();
// 	ImGui_ImplGlfw_NewFrame();
// 	igNewFrame();

// 	igBegin("Test", NULL, 0);
// 	igInputInt3("Position:", position, 0);
// 	igText(
// 			"Cube: %i:%i",
// 			chunks[0].chunk[position[0]][position[1]][position[2]][0],
// 			chunks[0].chunk[position[0]][position[1]][position[2]][1]);
// 	igText("Camera = x: %f, y: %f, z: %f", cameraPos[0], cameraPos[1], cameraPos[2]);
// 	igText("Camera chunk = x: %i, y: %i", cameraChunk[0], cameraChunk[1]);
// 	if (igButton("generate", (ImVec2){100, 30}))
// 	{
// 		atualizeMovement();
// 	}
// 	igEnd();

// 	gui_render(window);

// #ifdef IMGUI_HAS_DOCK
// 	if (ioptr->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
// 	{
// 		GLFWwindow *backup_current_window = glfwGetCurrentContext();
// 		igUpdatePlatformWindows();
// 		igRenderPlatformWindowsDefault(NULL, NULL);
// 		glfwMakeContextCurrent(backup_current_window);
// 	}
// #endif
// }

INCBIN(box, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_box.png");
INCBIN(iron_frame, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_iron_frame.png");
INCBIN(stone, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_stone.png");
INCBIN(grass, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_grass.png");
INCBIN(dirt_with_grass, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_dirt_with_grass.png");
INCBIN(dirt, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_dirt.png");
INCBIN(sand, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_sand.png");
INCBIN(oak_leaves, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_oak_leaves.png");
INCBIN(glass, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_glass.png");
INCBIN(water, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_water.png");
INCBIN(oak_log, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_oak_tree.png");
INCBIN(oak_log_top, "V:/projetos/apisOrLibs/opengl/vulpescraft1/src/textures/default_oak_tree_top.png");

#define LOAD_TEXTURE(name) \
	texturesData[textureCount++] = stbi_load_from_memory((const stbi_uc *)&incbin_ ## name ## _start, (char*)&incbin_ ## name ## _end - (char*)&incbin_ ## name ## _start, &width, &height, &comp, 0);

color calcColor(unsigned int textureId)
{
	unsigned char *texture = texturesData[textureId];
	long int total = 0;

	long int totalR = 0;
	long int totalG = 0;
	long int totalB = 0;

	for (int i = 0; i < 16 * 16 * 4; i += 4)
	{
		if (texture[i + 3])
		{
			++total;
			totalR += (int)texture[i];
			totalG += (int)texture[i + 1];
			totalB += (int)texture[i + 2];
		}
	}

	return (color){
		.r = (float)totalR / total,
		.g = (float)totalG / total,
		.b = (float)totalB / total
	};
}

#define SEC_TO_NS(sec) ((sec)*1000000000)

uint64_t nanoseconds;
struct timespec ts_init;
struct timespec ts_end;

void init(void)
{
	timespec_get(&ts_init, TIME_UTC);
	chunks = (chunkNode *)malloc(chunkLimit * sizeof(chunkNode));
	
	stbi_set_flip_vertically_on_load(true);
	int width, height, comp;
	LOAD_TEXTURE(box)	            // 0
	LOAD_TEXTURE(iron_frame)      // 1
	LOAD_TEXTURE(stone)           // 2
	LOAD_TEXTURE(grass)           // 3
	LOAD_TEXTURE(dirt_with_grass) // 4
	LOAD_TEXTURE(dirt)            // 5
	LOAD_TEXTURE(sand)            // 6
	LOAD_TEXTURE(oak_leaves)      // 7
	LOAD_TEXTURE(glass)           // 8
	LOAD_TEXTURE(water)           // 9
	LOAD_TEXTURE(oak_log)         // 10
	LOAD_TEXTURE(oak_log_top)     // 11

	blocks[1][0] = (block){.name = "Stone", .type = solid, .textures = {2, 2, 2, 2, 2, 2}, .averageColor = calcColor(2)};
	blocks[2][0] = (block){.name = "Dirt", .type = solid, .textures = {5, 5, 5, 5, 5, 5}, .averageColor = calcColor(5)};
	blocks[2][1] = (block){.name = "Dirt with grass", .type = solid, .textures = {3, 5, 4, 4, 4, 4}, .averageColor = calcColor(3)};
	blocks[3][0] = (block){.name = "Oak leaves", .type = empty, .textures = {7, 7, 7, 7, 7, 7}, .averageColor = calcColor(7)};
	blocks[4][0] = (block){.name = "Glass", .type = transparent, .textures = {8, 8, 8, 8, 8, 8}, .averageColor = calcColor(8)};
	blocks[5][0] = (block){.name = "Water source", .type = transparent, .textures = {9, 9, 9, 9, 9, 9}, .averageColor = calcColor(9)};
	blocks[6][0] = (block){.name = "Sand", .type = solid, .textures = {6, 6, 6, 6, 6, 6}, .averageColor = calcColor(6)};
	blocks[7][0] = (block){.name = "Sand", .type = solid, .textures = {11, 11, 10, 10, 10, 10}, .averageColor = calcColor(11)};
	
	cameraChunk[0] = (int)floor(cameraPos[0] / 16);
	cameraChunk[1] = (int)floor(cameraPos[2] / 16);
	cameraChunkPos[0] = cameraPos[0] - cameraChunk[0] * 16;
	cameraChunkPos[1] = cameraPos[1];
	cameraChunkPos[2] = cameraPos[2] - cameraChunk[1] * 16;
	glm_ivec2_copy(cameraChunk, lastChunk);

	generateManyChunks(&overworld, cameraChunk[0], cameraChunk[1], &chunks, &chunkCount, &chunkLimit);
	generateChunkSides(&overworld, chunks, 0, chunkCount);
	generateMesh(chunks, 0, chunkCount);
	generateVertices(chunks, 0, chunkCount);
	generateChunksMap(chunks, 0, chunkCount);
	generateMiniMap(&overworld, 0, 0);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	for (int i = 0; i < chunkCount; i++)
	{
		chunkNode *c = &chunks[i];
		addVBO(c);
		chunksToView[chunksTVCount++] = c->index;
	}

	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &lightCasterVShader, NULL);
	glCompileShader(vertexShader);

	int success;
	char infoLog[512];

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
	}

	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragmentShader, 1, &lightCasterFShader, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
	}

	loadShader(vertexShader, fragmentShader);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT); // set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	void *data = malloc(4 * 16 * 16 * textureCount);
	for (size_t i = 0; i < textureCount; i++)
	{
		memcpy(data + 16 * 16 * i * 4, texturesData[i], 16 * 16 * 4);
		stbi_image_free(texturesData[i]);
	}

	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 16, 16, textureCount, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
	free(data);

	glUseProgram(shaderProgram[0]);
	glUniform1i(glGetUniformLocation(shaderProgram[0], "textures"), 0);

	uniformView = glGetUniformLocation(shaderProgram[0], "view");
	uniformProjection = glGetUniformLocation(shaderProgram[0], "projection");
	uniformModel = glGetUniformLocation(shaderProgram[0], "model");

	unsigned int uniformLoc = glGetUniformLocation(shaderProgram[0], "light.direction");
	glUniform3f(uniformLoc, -0.2f, -1.0f, -0.3f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "viewPos");
	glUniform3f(uniformLoc, cameraChunkPos[0], cameraChunkPos[1], cameraChunkPos[2]);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.ambient");
	glUniform3f(uniformLoc, 0.5f, 0.5f, 0.5f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.diffuse");
	glUniform3f(uniformLoc, 0.8f, 0.8f, 0.8f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.specular");
	glUniform3f(uniformLoc, 1.0f, 1.0f, 1.0f);

	uniformLoc = glGetUniformLocation(shaderProgram[0], "material.shininess");
	glUniform1f(uniformLoc, 32.0f);

	uniformLoc = glGetUniformLocation(shaderProgram[0], "model");
	mat4 model = GLM_MAT4_IDENTITY_INIT;
	glUniformMatrix4fv(uniformLoc, 1, false, (float *)model);

	glClearColor(0.6666f, 0.8156f, 0.9921f, 1.0f);
	// vec4(.6666, .8156, .9921, 1.)

	glm_translate(view, (vec3){0.0f, 0.0f, -3.0f});
	glm_perspective(glm_rad(fov), (float)screenWidth / (float)screenHeight, near, far, projection);
	vec3 center;
	glm_vec3_add(cameraChunkPos, cameraFront, center);
	glm_lookat(cameraChunkPos, center, cameraUp, view);
	calculateFrustum();

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	float vertices[] = {
        // positions          // texture coords
         1.f,  1.f, -1.0f,   1.0f, 1.0f, // top right
         1.f, -1.f, -1.0f,   1.0f, 0.0f, // bottom right
        -1.f, -1.f, -1.0f,   0.0f, 0.0f, // bottom left
        -1.f,  1.f, -1.0f,   0.0f, 1.0f  // top left 
    };
	unsigned int indices[] = {
			0, 2, 1, // first Triangle
			0, 3, 2	 // second Triangle
	};

	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &mapVShader, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
	}

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragmentShader, 1, &mapFShader, NULL);
	glCompileShader(fragmentShader);
	
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
	}

	loadShader(vertexShader, fragmentShader);
	glUseProgram(shaderProgram[1]);
	uniformRealPos = glGetUniformLocation(shaderProgram[1], "realPos");
	uniformOffset = glGetUniformLocation(shaderProgram[1], "offset");
	uniformMapScale = glGetUniformLocation(shaderProgram[1], "scale");
	uniformCameraDirection = glGetUniformLocation(shaderProgram[1], "direction");
	glUniform3f(uniformMapScale, minimapScale, (float)screenWidth / screenHeight * minimapScale, 0);
	cameraDirection = atan2(cameraFront[2], cameraFront[0]);
	glUniform1f(uniformCameraDirection, cameraDirection);
	glUniform3f(uniformOffset, (screenWidth - 300.) / screenWidth, (screenHeight - 300.) / screenHeight, 0);
	cameraChunkPos[0] = cameraPos[0] - cameraChunk[0] * 16;
	cameraChunkPos[1] = cameraPos[1];
	cameraChunkPos[2] = cameraPos[2] - cameraChunk[1] * 16;
	float posX = 2 * cameraChunkPos[2] / (minimapScale * screenWidth);
	float posY = 2 * cameraChunkPos[0] / (minimapScale * screenWidth);
	glUniform3f(uniformRealPos, posX, posY, 0);

	glGenVertexArrays(1, &mapVAO);
	glGenBuffers(1, &mapVBO);
	glGenBuffers(1, &mapEBO);

	glBindVertexArray(mapVAO);

	glBindBuffer(GL_ARRAY_BUFFER, mapVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mapEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// screen color
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);


	float quadVertices[] = {
		// positions   // texCoords
		-1.0f,  1.0f,  0.0f, 1.0f,
		-1.0f, -1.0f,  0.0f, 0.0f,
		1.0f, -1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f,  0.0f, 1.0f,
		1.0f, -1.0f,  1.0f, 0.0f,
		1.0f,  1.0f,  1.0f, 1.0f
	};

	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glGenTextures(1, &textureColorbuffer);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screenWidth, screenHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
	
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &depthBuffer);
	glBindTexture(GL_TEXTURE_2D, depthBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, screenWidth, screenHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	float f4[4] = {1.f, 1.f, 1.f, 1.f};
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, f4);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthBuffer, 0);

	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		puts("ERROR::FRAMEBUFFER:: Framebuffer is not complete!");

	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);
	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));


	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &vertexShaderSrc, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
	}

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragmentShader, 1, &fragmentShaderSrc, NULL);
	glCompileShader(fragmentShader);
	
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
	}

	loadShader(vertexShader, fragmentShader);


	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	glm_mat4_inv(projection, invProjection);
	glm_mat4_inv(view, invView);
	uniformLoc = glGetUniformLocation(shaderProgram[2], "screenTexture");
	glUniform1i(uniformLoc, 0);
	uniformLoc = glGetUniformLocation(shaderProgram[2], "depth");
	glUniform1i(uniformLoc, 1);
	uniformLoc = glGetUniformLocation(shaderProgram[2], "sun");
	glUniform3f(uniformLoc, -0.2f, -1.0f, -0.3f);
	uniformLoc = glGetUniformLocation(shaderProgram[2], "fov");
	glUniform1f(uniformLoc, glm_rad(fov));
	uniformScreen = glGetUniformLocation(shaderProgram[2], "screen");
	glUniform2f(uniformScreen, screenWidth, screenHeight);
	uniformInvView = glGetUniformLocation(shaderProgram[2], "invViewMatrix");
	glUniformMatrix4fv(uniformInvView, 1, false, (float *)invView);
	uniformPers = glGetUniformLocation(shaderProgram[2], "persMatrix");
	glUniformMatrix4fv(uniformPers, 1, false, (float *)projection);
	uniformInvPers = glGetUniformLocation(shaderProgram[2], "invPersMatrix");
	glUniformMatrix4fv(uniformInvPers, 1, false, (float *)invProjection);
	uniformViewPos = glGetUniformLocation(shaderProgram[2], "viewPos");
	glUniform3f(uniformViewPos, cameraPos[0], cameraPos[1], cameraPos[2]);
	uniformView2 = glGetUniformLocation(shaderProgram[2], "viewMatrix");
	glUniformMatrix4fv(uniformView2, 1, false, (float *)view);
	uniformWater = glGetUniformLocation(shaderProgram[2], "water");
	glUniform1i(uniformWater, 0);
	
	
	glShaderSource(vertexShader, 1, &mapsVShaderSrc, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
	}

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragmentShader, 1, &mapsFShaderSrc, NULL);
	glCompileShader(fragmentShader);
	
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
	}

	loadShader(vertexShader, fragmentShader);

	glGenVertexArrays(1, &worldVAO);
	glGenBuffers(1, &worldVBO);
	glGenBuffers(1, &worldEBO);

	glBindVertexArray(worldVAO);

	glBindBuffer(GL_ARRAY_BUFFER, worldVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, worldEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	uniformWorldMapScale = glGetUniformLocation(shaderProgram[3], "scale");
	float scale = 50. / screenWidth;
	glUniform2f(uniformWorldMapScale, scale, (float)screenWidth / screenHeight * scale);
	uniformWorldMapOffset = glGetUniformLocation(shaderProgram[3], "offset");
	glUniform2f(uniformWorldMapOffset, 0, 0);
	uniformWorldMapScreen = glGetUniformLocation(shaderProgram[3], "screen");
	glUniform2f(uniformWorldMapScreen, screenWidth, screenHeight);
	uniformWorldMapPos = glGetUniformLocation(shaderProgram[3], "position");
	glUniform2f(uniformWorldMapPos, 0, 0);

	
	printf("%i\n", glGetError());
	
	timespec_get(&ts_end, TIME_UTC);

	nanoseconds = SEC_TO_NS((uint64_t)ts_end.tv_sec) + (uint64_t)ts_end.tv_nsec - SEC_TO_NS((uint64_t)ts_init.tv_sec) + (uint64_t)ts_init.tv_nsec;

	printf("Tempo para gerar as chunks: %llu\n", nanoseconds);
}

double currentTime;

void render(void)
{
	currentTime = glfwGetTime();

	if (mapView)
	{
		glUseProgram(shaderProgram[3]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		for (int i = 0; i < chunkCount; i++)
		{
			glUniform2f(uniformWorldMapOffset, chunks[i].posZ, chunks[i].posX);
			glBindTexture(GL_TEXTURE_2D, chunks[i].texture);
			glBindVertexArray(worldVAO);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
		}
	}
	else
	{
		vec3 center;
		glm_vec3_add(cameraChunkPos, cameraFront, center);
		glm_lookat(cameraChunkPos, center, cameraUp, view);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glEnable(GL_DEPTH_TEST);

		glUseProgram(shaderProgram[0]);

		glUniformMatrix4fv(uniformView, 1, false, (float *)view);
		glUniformMatrix4fv(uniformProjection, 1, false, (float *)projection);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, texture);

		mat4 model = GLM_MAT4_IDENTITY_INIT;

		for (int i = 0; i <= chunksIFCount; i++)
		{
			chunkNode *c = &chunks[chunksInFront[i]];
			vec3 translate = {(c->posX - cameraChunk[0]) * 16, 0, (c->posZ - cameraChunk[1]) * 16};

			glBindVertexArray(c->VAO);
			glm_translate(model, translate);
			glUniformMatrix4fv(uniformModel, 1, false, (float *)model);
			glm_mat4_identity(model);
			if (c->waterIndex >= 0)
			{
				glDrawElements(GL_TRIANGLES, c->waterIndex, GL_UNSIGNED_INT, 0);
			}
			else
			{
				glDrawElements(GL_TRIANGLES, c->indicesBufferCount, GL_UNSIGNED_INT, 0);
			}
			glBindVertexArray(0);
		}

		for (int i = 0; i <= chunksIFCount; i++)
		{
			chunkNode *c = &chunks[chunksInFront[i]];
			vec3 translate = {(c->posX - cameraChunk[0]) * 16, 0, (c->posZ - cameraChunk[1]) * 16};

			glBindVertexArray(c->VAO);
			glm_translate(model, translate);
			glUniformMatrix4fv(uniformModel, 1, false, (float *)model);
			glm_mat4_identity(model);
			if (c->waterIndex >= 0)
			{
				glDrawElements(GL_TRIANGLES, (c->indicesBufferCount - c->waterIndex) + 1, GL_UNSIGNED_INT, (GLvoid*) (sizeof(int)*c->waterIndex));
			}
				
			glBindVertexArray(0);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shaderProgram[2]);
		glBindVertexArray(quadVAO);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, depthBuffer);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glUniform1f(uniformPitch, glm_rad(pitch));
		glUniform1f(uniformYaw, glm_rad(yaw));
		glm_mat4_inv(view, invView);
		glUniformMatrix4fv(uniformView2, 1, false, (float *)view);
		glUniformMatrix4fv(uniformInvView, 1, false, (float *)invView);
		glUniformMatrix4fv(uniformPers, 1, false, (float *)projection);
		glUniform3f(uniformViewPos, cameraChunkPos[0], cameraChunkPos[1], cameraChunkPos[2]);

		glUseProgram(shaderProgram[1]);
		// glClear(GL_DEPTH_BUFFER_BIT);

		glActiveTexture(GL_TEXTURE0);

		glBindTexture(GL_TEXTURE_2D, megaMapTexture);
		glBindVertexArray(mapVAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		if (walked)
		{
			atualizeMovement();
			calculateFrustum();
			float posX = cameraChunkPos[2] / (minimapScale * screenWidth);
			float posY = cameraChunkPos[0] / (minimapScale * screenWidth);
			glUniform3f(uniformRealPos, posX, posY, 0);
			walked = false;

			chunkNode* c = &chunks[getChunkIdx(&overworld, cameraChunk[0], cameraChunk[1])];

			int x = floor(cameraChunkPos[0] + .5);
			int y = floor(cameraChunkPos[1] + .5);
			int z = floor(cameraChunkPos[2] + .5);

			if (x > 15)
			{
				c = &chunks[c->east];
				x = 0;
			}
			if (z > 15)
			{
				c = &chunks[c->north];
				z = 0;
			};

			glUseProgram(shaderProgram[2]);
			if (c->chunk[x][y][z][0] == 5)
			{
				if (!isOnWater)
				{
					glUniform1i(uniformWater, 1);
				}
				isOnWater = true;
			}
			else
			{
				if (isOnWater)
				{
					glUniform1i(uniformWater, 0);
				}
				isOnWater = false;
			}
		}
	}

	// fps counter
	++nbFrames;
	deltaTime = currentTime - lastFrame;
	lastFrame = currentTime;
	if (currentTime - lastTime >= 1.0)
	{
		fps = nbFrames;
		nbFrames = 0;
		lastTime += 1.0;
	}
}

int main(int argc, char **args)
{
	GLFWwindow *window;

	if (!glfwInit())
		return -1;

	window = glfwCreateWindow(screenWidth, screenHeight, "Textures test", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	glfwSetWindowSizeCallback(window, resizeCallback);
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetScrollCallback(window, scrollCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	if (!mapView)
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	init();
	// gui_init(window);
	while (!glfwWindowShouldClose(window))
	{
		render();
		glFlush();
		// gui_update(window);

		glfwSwapBuffers(window);
		glfwPollEvents();
		processInput(window);
		sprintf(title, "fps: %d", fps);
		glfwSetWindowTitle(window, title);
	}

	free(chunks);
	glfwDestroyWindow(window);
	glfwTerminate();
	// gui_terminate();
	
	return 0;
}